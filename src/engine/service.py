# src/engine/service.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import requests

from preprocess import FlowWindowDataset
from TCN_Transformer import TCNTransformerModel
from AutoEncoder import TCNAutoencoder
from .fusion_model import compute_ae_scores


@dataclass
class SplunkHECConfig:
    url: str                 # e.g. "http://192.168.8.129:8088/services/collector"
    token: str               # HEC token
    index: str = "main"
    sourcetype: str = "_json"
    host: str = "model-server"
    verify_tls: bool = False
    timeout_sec: int = 3


class FusionService:
    """
    - config/ckpt 로드 1회
    - parquet 리플레이로 윈도잉/추론/퓨전 수행
    - (옵션) Splunk HEC로 JSON 이벤트 전송
    """

    def __init__(
        self,
        root_dir: str | Path,
        config_rel: str = "processed/preprocess_config.json",
        tcn_ckpt_rel: str = "models/tcn_transformer_v2_best.pt",
        ae_ckpt_rel: str = "models/ae_tcn_best.pt",
        seq_len: int = 128,
        stride: int = 64,
        batch_size: int = 64,
        device: Optional[str] = None,
        ae_threshold: float = 1.5624,
        ae_agg_mode: str = "topk",
        ae_topk: int = 3,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.config_path = self.root_dir / config_rel
        self.tcn_ckpt = self.root_dir / tcn_ckpt_rel
        self.ae_ckpt = self.root_dir / ae_ckpt_rel

        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size

        self.ae_threshold = float(ae_threshold)
        self.ae_agg_mode = ae_agg_mode
        self.ae_topk = int(ae_topk)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- load config ----
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.numeric_cols = self.cfg["numeric_cols"]
        self.label_classes = self.cfg["label_classes"]

        numeric_dim_base = len(self.numeric_cols)
        window_feat_dim = 5
        self.numeric_dim = numeric_dim_base + window_feat_dim
        self.num_classes = len(self.label_classes)

        self.num_port_classes = self.cfg["other_port_idx"] + 1
        self.num_proto_classes = self.cfg["proto_other_idx"] + 1

        # ---- load models (fusion_test.py와 동일 하이퍼파라미터) ----
        self.tcn_model = TCNTransformerModel(
            numeric_dim=self.numeric_dim,
            num_classes=self.num_classes,
            num_port_classes=self.num_port_classes,
            num_proto_classes=self.num_proto_classes,
            d_model=128,
            num_heads=4,
            num_layers=2,
            tcn_channels=128,
            tcn_kernel_size=3,
            tcn_num_layers=2,
            dropout=0.1,
            max_len=self.seq_len + 1,
        ).to(self.device)

        state_tcn = torch.load(self.tcn_ckpt, map_location=self.device)
        self.tcn_model.load_state_dict(state_tcn)
        self.tcn_model.eval()

        self.ae_model = TCNAutoencoder(
            numeric_dim=self.numeric_dim,
            num_port_classes=self.num_port_classes,
            num_proto_classes=self.num_proto_classes,
            d_model=128,
            tcn_kernel_size=3,
            tcn_num_layers=3,
            dropout=0.1,
        ).to(self.device)

        state_ae = torch.load(self.ae_ckpt, map_location=self.device)
        self.ae_model.load_state_dict(state_ae)
        self.ae_model.eval()

        print("[FusionService] ready")
        print("  root_dir     :", self.root_dir)
        print("  config_path  :", self.config_path)
        print("  tcn_ckpt     :", self.tcn_ckpt)
        print("  ae_ckpt      :", self.ae_ckpt)
        print("  device       :", self.device)
        print("  seq_len/stride:", self.seq_len, self.stride)
        print("  ae_threshold :", self.ae_threshold)
        print("  label_classes:", self.label_classes)

    @torch.no_grad()
    def infer_parquet_and_send(
        self,
        parquet_rel: str = "processed/flows_test_scaled.parquet",
        splunk: Optional[SplunkHECConfig] = None,
        max_batches: Optional[int] = 10,
        sleep_sec: float = 0.0,
    ) -> Dict[str, Any]:
        """
        parquet를 FlowWindowDataset으로 윈도잉 → 배치 추론 → (옵션) Splunk HEC 전송

        return: 요약 통계
        """
        parquet_path = (self.root_dir / parquet_rel).resolve()

        ds = FlowWindowDataset(
            parquet_path=str(parquet_path),
            config_path=str(self.config_path),
            seq_len=self.seq_len,
            stride=self.stride,
        )

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        total_windows = 0
        sent_events = 0
        tcn_pred_attack = 0
        fusion_pred_attack = 0

        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break

            numeric, cat, mask, y = batch
            numeric = numeric.to(self.device)
            cat = cat.to(self.device)
            mask = mask.to(self.device)

            # --- TCN ---
            logits = self.tcn_model(numeric, cat, mask)   # [B, C]
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).bool()                 # [B, C]
            pred_attack_any = preds[:, 1:].any(dim=1)      # [B]

            # --- AE (TCN이 놓친 것만) ---
            # 여기선 GT(y)를 굳이 쓰지 않고 "TCN이 benign이라 본 것 중 AE 이상"을 추가 탐지로 사용
            idx_missed = torch.nonzero(~pred_attack_any, as_tuple=True)[0]  # [N_missed]

            ae_recovered_mask = torch.zeros_like(pred_attack_any)
            ae_scores_all = torch.full((pred_attack_any.shape[0],), float("nan"), device=self.device)

            if idx_missed.numel() > 0:
                recon = self.ae_model(numeric[idx_missed], cat[idx_missed], mask[idx_missed])  # [N, L, F]
                scores = compute_ae_scores(
                    recon=recon,
                    target=numeric[idx_missed],
                    mask=mask[idx_missed],
                    agg_mode=self.ae_agg_mode,
                    topk=self.ae_topk,
                )  # [N]
                ae_anom = scores >= self.ae_threshold
                ae_recovered_mask[idx_missed] = ae_anom
                ae_scores_all[idx_missed] = scores

            fusion_attack_any = pred_attack_any | ae_recovered_mask

            # ---- stats ----
            B = numeric.shape[0]
            total_windows += B
            tcn_pred_attack += int(pred_attack_any.sum().item())
            fusion_pred_attack += int(fusion_attack_any.sum().item())

            # ---- send to Splunk ----
            if splunk is not None:
                # per-window event 생성
                # (window_id는 단순 카운터로; 나중에 dst_ip/time을 window feature에서 넣어도 됨)
                payloads = []
                for i in range(B):
                    event = {
                        "window_id": f"b{b_idx}_i{i}",
                        "pred": {
                            "tcn_attack_any": bool(pred_attack_any[i].item()),
                            "fusion_attack_any": bool(fusion_attack_any[i].item()),
                            "ae_score": None if torch.isnan(ae_scores_all[i]) else float(ae_scores_all[i].item()),
                        },
                        "meta": {
                            "ae_threshold": self.ae_threshold,
                            "agg_mode": self.ae_agg_mode,
                            "topk": self.ae_topk,
                            "seq_len": self.seq_len,
                            "stride": self.stride,
                        },
                    }
                    payloads.append({
                        "event": event,
                        "host": splunk.host,
                        "sourcetype": splunk.sourcetype,
                        "index": splunk.index,
                        "time": time.time(),
                    })

                sent = self._send_hec_batch(splunk, payloads)
                sent_events += sent

            if sleep_sec > 0:
                time.sleep(sleep_sec)

            print(f"[batch {b_idx}] windows={B} tcn_attack={int(pred_attack_any.sum())} fusion_attack={int(fusion_attack_any.sum())} sent={sent_events}")

        return {
            "total_windows": total_windows,
            "tcn_pred_attack_windows": tcn_pred_attack,
            "fusion_pred_attack_windows": fusion_pred_attack,
            "sent_events": sent_events,
            "max_batches": max_batches,
        }

    def _send_hec_batch(self, splunk: SplunkHECConfig, payloads: List[Dict[str, Any]]) -> int:
        headers = {
            "Authorization": f"Splunk {splunk.token}",
            "Content-Type": "application/json",
        }
        ok = 0
        for p in payloads:
            try:
                r = requests.post(
                    splunk.url,
                    headers=headers,
                    json=p,
                    timeout=splunk.timeout_sec,
                    verify=splunk.verify_tls,
                )
                if 200 <= r.status_code < 300:
                    ok += 1
            except Exception:
                pass
        return ok
