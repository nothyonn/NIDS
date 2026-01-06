# src/engine/service.py
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from TCN_Transformer import TCNTransformerModel
from AutoEncoder import TCNAutoencoder

from .fusion_model import compute_ae_scores
from .online_preprocess import OnlinePreprocessor
from .online_window import OnlineWindowBuffer, WindowItem
from .hec_client import SplunkHECClient, SplunkHECConfig


def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class FusionService:
    """
    - (online) ingest(JSON flows) -> preprocess -> dst_ip windowing -> infer -> (opt) Splunk HEC
    - (offline) replay(parquet) 는 별도 유지하고 싶으면 다른 모듈에서 돌려도 됨
      (지금 파일은 online 중심)
    """

    def __init__(
        self,
        root_dir: str | Path,
        config_rel: str = "processed/preprocess_config.json",
        tcn_ckpt_rel: str = "models/tcn_transformer_v2_best.pt",
        ae_ckpt_rel: str = "models/ae_tcn_best.pt",
        seq_len: int = 128,
        stride: int = 64,
        device: Optional[str] = None,
        ae_threshold: float = 1.5624,
        ae_agg_mode: str = "topk",
        ae_topk: int = 3,
        max_buffer_per_dst: int = 5000,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.config_path = (self.root_dir / config_rel).resolve()
        self.tcn_ckpt = (self.root_dir / tcn_ckpt_rel).resolve()
        self.ae_ckpt = (self.root_dir / ae_ckpt_rel).resolve()

        self.seq_len = int(seq_len)
        self.stride = int(stride)

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
        self.num_classes = len(self.label_classes)

        numeric_dim_base = len(self.numeric_cols)  # ex) 77
        window_feat_dim = 5
        self.numeric_dim = numeric_dim_base + window_feat_dim

        self.num_port_classes = int(self.cfg["other_port_idx"]) + 1
        self.num_proto_classes = int(self.cfg["proto_other_idx"]) + 1

        # ---- online components ----
        self.prep = OnlinePreprocessor(self.config_path)
        self.winbuf = OnlineWindowBuffer(
            seq_len=self.seq_len,
            stride=self.stride,
            max_buffer=max_buffer_per_dst,
        )

        # ---- models ----
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
        self.tcn_model.load_state_dict(torch.load(self.tcn_ckpt, map_location=self.device))
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
        self.ae_model.load_state_dict(torch.load(self.ae_ckpt, map_location=self.device))
        self.ae_model.eval()

        print("[FusionService] ready")
        print("  device        :", self.device)
        print("  seq_len/stride:", self.seq_len, self.stride)

    @torch.no_grad()
    def ingest_flows_and_send(
        self,
        flows: List[Dict[str, Any]],
        *,
        splunk_cfg: Optional[SplunkHECConfig],
        drop_label: bool = True,
        force_flush: bool = False,
        min_flush_len: int = 16,
        max_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        flows(JSON list) -> preprocess -> buffer -> pop windows -> infer -> (opt) hec send
        """
        if not flows:
            return {"ok": False, "error": "empty flows"}

        # 1) preprocess (모델에서 처리)
        rows = self.prep.transform(flows, drop_label=drop_label)

        # 2) buffer add
        self.winbuf.add_flows(rows)

        # 3) pop windows (dst_ip 기준)
        windows: List[WindowItem] = self.winbuf.pop_windows(
            force_flush=force_flush,
            min_flush_len=min_flush_len,
        )
        if max_windows is not None:
            windows = windows[:max_windows]

        hec = SplunkHECClient(splunk_cfg) if splunk_cfg is not None else None

        popped_windows = 0
        sent_events = 0
        tcn_attack_windows = 0
        ae_attack_windows = 0

        events: List[Dict[str, Any]] = []

        for w in windows:
            numeric, cat, mask = self._build_window_tensors(w)

            # ---- TCN ----
            logits = self.tcn_model(numeric, cat, mask)   # [1, C]
            probs = torch.sigmoid(logits)[0]              # [C]
            preds = (probs >= 0.5)                        # [C]

            tcn_attack_classes = [
                self.label_classes[j]
                for j in range(1, self.num_classes)
                if bool(preds[j].item())
            ]
            tcn_attack_any = len(tcn_attack_classes) > 0

            # ---- AE (TCN benign일 때만) ----
            ae_used = False
            ae_score = None
            ae_attack = False

            if not tcn_attack_any:
                recon = self.ae_model(numeric, cat, mask)  # [1, L, F]
                scores = compute_ae_scores(
                    recon=recon,
                    target=numeric,
                    mask=mask,
                    agg_mode=self.ae_agg_mode,
                    topk=self.ae_topk,
                )
                ae_score = float(scores[0].item())
                ae_used = True
                ae_attack = ae_score >= self.ae_threshold

            # ---- decision ----
            if tcn_attack_any:
                decision = {
                    "source": "TCN",
                    "attack": True,
                    "attack_types": tcn_attack_classes,
                    "confidence": {
                        self.label_classes[j]: float(probs[j].item())
                        for j in range(1, self.num_classes)
                        if bool(preds[j].item())
                    },
                }
                tcn_attack_windows += 1

            elif ae_attack:
                decision = {
                    "source": "AE",
                    "attack": True,
                    "attack_types": ["ANOMALY"],
                    "ae_score": ae_score,
                    "ae_threshold": self.ae_threshold,
                    "agg_mode": self.ae_agg_mode,
                    "topk": self.ae_topk,
                }
                ae_attack_windows += 1

            else:
                decision = {"source": "NONE", "attack": False, "attack_types": []}

            event = {
                "dst_ip": w.dst_ip,
                "real_len": int(w.real_len),
                "decision": decision,
                "meta": {
                    "seq_len": self.seq_len,
                    "stride": self.stride,
                    "ae_used": ae_used,
                    "ts": time.time(),
                },
            }
            events.append(event)
            popped_windows += 1

        if hec is not None and events:
            sent_events = hec.send_events(events)

        return {
            "ok": True,
            "input_flows": len(flows),
            "popped_windows": popped_windows,
            "tcn_attack_windows": tcn_attack_windows,
            "ae_attack_windows": ae_attack_windows,
            "sent_events": sent_events,
        }

    def _build_window_tensors(self, w: WindowItem) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        WindowItem(rows, real_len) -> (numeric, cat, mask)
        numeric: [1, L, 77+5]
        cat   : [1, L, 3]  (sport_idx, dport_idx, proto_idx)
        mask  : [1, L]
        """
        rows = w.rows
        real_len = int(w.real_len)
        L = self.seq_len

        # base numeric [real_len, 77]
        base = np.zeros((real_len, len(self.numeric_cols)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(self.numeric_cols):
                base[i, j] = float(r.get(c, 0.0))

        # ----- window-level features (5) -----
        total_flows = float(real_len)

        src_vals = [r.get("Source IP") for r in rows if r.get("Source IP") is not None]
        if src_vals:
            vc = pd.Series(src_vals).value_counts()
            unique_src = float(len(vc))
            counts = vc.to_numpy(dtype=np.float32)
        else:
            unique_src = 1.0
            counts = np.array([], dtype=np.float32)

        avg_flows_per_src = total_flows / unique_src if unique_src > 0 else 0.0
        src_ent_raw = _shannon_entropy(counts) if unique_src > 0 and len(counts) > 0 else 0.0

        ts_list = [r.get("Timestamp") for r in rows if r.get("Timestamp") is not None]
        if ts_list:
            t_min = min(ts_list)
            t_max = max(ts_list)
            window_duration = float(max((t_max - t_min).total_seconds(), 1e-6))
        else:
            window_duration = 1e-6

        flow_rate = total_flows / window_duration

        f1 = np.log1p(unique_src) / 5.0
        f2 = np.log1p(avg_flows_per_src) / 5.0
        if unique_src > 1:
            max_ent = math.log2(unique_src)
            f3 = float(np.clip(src_ent_raw / (max_ent + 1e-12), 0.0, 1.0))
        else:
            f3 = 0.0
        f4 = np.log1p(window_duration) / 10.0
        f5 = np.log1p(flow_rate) / 5.0

        win_feat = np.array([f1, f2, f3, f4, f5], dtype=np.float32)
        win_feat_real = np.repeat(win_feat[None, :], real_len, axis=0)  # [real_len,5]
        numeric_real = np.concatenate([base, win_feat_real], axis=1)    # [real_len,82]

        # pad numeric to [L, 82]
        numeric = np.zeros((L, numeric_real.shape[1]), dtype=np.float32)
        numeric[:real_len] = numeric_real

        # categorical [L,3]
        cat = np.zeros((L, 3), dtype=np.int64)
        for i, r in enumerate(rows):
            cat[i, 0] = int(r.get("sport_idx", 0))
            cat[i, 1] = int(r.get("dport_idx", 0))
            cat[i, 2] = int(r.get("proto_idx", 0))

        # mask [L]
        mask = np.zeros((L,), dtype=np.float32)
        mask[:real_len] = 1.0

        numeric_t = torch.tensor(numeric[None, :, :], dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(cat[None, :, :], dtype=torch.int64, device=self.device)
        mask_t = torch.tensor(mask[None, :], dtype=torch.float32, device=self.device)

        return numeric_t, cat_t, mask_t
