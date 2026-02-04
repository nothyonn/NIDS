# src/engine/service.py
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _norm_col(c: str) -> str:
    if c is None:
        return ""
    c = str(c).replace("\ufeff", "").strip()
    c = " ".join(c.split())
    return c


class FusionService:
    def __init__(
        self,
        root_dir: str | Path,
        config_rel: str = "processed/preprocess_config.json",
        tcn_ckpt_rel: str = "models/tcn_transformer_v2_best.pt",
        ae_ckpt_rel: str = "models/ae_tcn_best.pt",
        seq_len: int = 128,
        stride: int = 64,
        device: Optional[str] = None,
        ae_threshold: float = 1.5624,   # top3 p95
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

        # ---- debug settings (env로 제어) ----
        self.debug_enabled = os.getenv("MODEL_DEBUG", "1") == "1"
        self.debug_windows = int(os.getenv("MODEL_DEBUG_WINDOWS", "3"))
        self.splunk_debug_meta = os.getenv("SPLUNK_DEBUG_META", "1") == "1"
        self.sort_by_timestamp = os.getenv("SORT_ROWS_BY_TIMESTAMP", "1") == "1"
        self.send_dropped_windows = os.getenv("SEND_DROPPED_WINDOWS", "0") == "1"

        # 학습 규칙: 패딩 과다 윈도우 drop
        self.min_real_ratio = float(os.getenv("MIN_REAL_RATIO", "0.3"))
        self.min_real_len = max(1, int(math.ceil(self.seq_len * self.min_real_ratio)))

        self.model_api_log = Path(os.getenv("MODEL_API_LOG", "/data/log/model_api.log"))
        self.model_api_log.parent.mkdir(parents=True, exist_ok=True)

        # ---- load config ----
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.numeric_cols = self.cfg["numeric_cols"]
        self.label_classes = self.cfg["label_classes"]
        self.num_classes = len(self.label_classes)

        numeric_dim_base = len(self.numeric_cols)
        window_feat_dim = 5
        self.numeric_dim = numeric_dim_base + window_feat_dim

        self.other_port_idx = int(self.cfg["other_port_idx"])
        self.other_proto_idx = int(self.cfg["proto_other_idx"])
        self.num_port_classes = self.other_port_idx + 1
        self.num_proto_classes = self.other_proto_idx + 1

        self.config_sha256 = _sha256_file(self.config_path)
        self.numeric_cols_hash = _sha256_text("\n".join(self.numeric_cols))
        self.schema_id = self.config_sha256[:12]
        self.model_version = os.getenv("MODEL_VERSION", self.tcn_ckpt.name)

        # ---- online components ----
        self.prep = OnlinePreprocessor(self.config_path)
        self.winbuf = OnlineWindowBuffer(seq_len=self.seq_len, stride=self.stride, max_buffer=max_buffer_per_dst)

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
        print("  schema_id     :", self.schema_id)
        print("  model_version :", self.model_version)

    # ---------------------------
    # schema / debug utils
    # ---------------------------
    def get_schema(self) -> Dict[str, Any]:
        # ✅ app.py의 /schema가 이걸 호출함 (없어서 AttributeError 났던 것)
        return {
            "schema_id": self.schema_id,
            "config_sha256": self.config_sha256,
            "numeric_cols_hash": self.numeric_cols_hash,
            "numeric_cols": self.numeric_cols,
            "label_classes": self.label_classes,
            "other_port_idx": self.other_port_idx,
            "other_proto_idx": self.other_proto_idx,
            "seq_len": self.seq_len,
            "stride": self.stride,
            "min_real_len": self.min_real_len,
            "min_real_ratio": self.min_real_ratio,
            "model_version": self.model_version,
            "ae_threshold": self.ae_threshold,
            "ae_agg_mode": self.ae_agg_mode,
            "ae_topk": self.ae_topk,
        }

    def _log_debug(self, kind: str, **fields: Any) -> None:
        if not self.debug_enabled:
            return
        rec = {"ts": time.time(), "kind": kind, "schema_id": self.schema_id, "model_version": self.model_version, **fields}
        try:
            with self.model_api_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _schema_coverage(self, row_like: Dict[str, Any]) -> Dict[str, Any]:
        keys = list(row_like.keys())
        strict_set = set(keys)

        norm_keys = {_norm_col(k) for k in keys}
        norm_numeric = {_norm_col(c) for c in self.numeric_cols}

        strict_inter = strict_set & set(self.numeric_cols)
        norm_inter = norm_keys & norm_numeric
        missing_norm = sorted(list(norm_numeric - norm_keys))

        cov_strict = len(strict_inter) / len(self.numeric_cols) if self.numeric_cols else 0.0
        cov_norm = len(norm_inter) / len(self.numeric_cols) if self.numeric_cols else 0.0

        return {
            "cov_strict": float(cov_strict),
            "cov_norm": float(cov_norm),
            "missing_norm_n": int(len(missing_norm)),
            "missing_norm_top": missing_norm[:20],
        }

    @torch.no_grad()
    def ingest_flows_and_send(
        self,
        flows: List[Dict[str, Any]],
        *,
        request_id: Optional[str] = None,
        splunk_cfg: Optional[SplunkHECConfig] = None,
        drop_label: bool = True,
        force_flush: bool = False,
        min_flush_len: int = 39,
        max_windows: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not flows:
            return {"ok": False, "error": "empty flows"}

        if request_id is None:
            request_id = str(flows[0].get("_debug_request_id") or uuid.uuid4())

        # (A) raw 기준 커버리지
        raw_cov = self._schema_coverage(flows[0])
        self._log_debug("ingest_begin", request_id=request_id, input_flows=len(flows), **raw_cov)

        # (B) rename/drop만 적용한 뒤 커버리지 (rename 맞는지 핵심 체크)
        renamed0 = self.prep.rename_and_clean_only(flows[0], drop_label=drop_label)
        ren_cov = self._schema_coverage(renamed0)
        self._log_debug("after_rename_schema", request_id=request_id, **ren_cov)

        # 1) preprocess (학습 규칙 재현 + row단 debug)
        rows = self.prep.transform(flows, drop_label=drop_label, attach_debug=True)

        # 2) buffer add
        self.winbuf.add_flows(rows)

        # 3) pop windows
        windows: List[WindowItem] = self.winbuf.pop_windows(force_flush=force_flush, min_flush_len=min_flush_len)
        if max_windows is not None:
            windows = windows[:max_windows]

        hec = SplunkHECClient(splunk_cfg) if splunk_cfg is not None else None

        popped_windows = 0
        sent_events = 0
        tcn_attack_windows = 0
        ae_attack_windows = 0
        dropped_windows = 0

        events: List[Dict[str, Any]] = []

        for wi, w in enumerate(windows):
            popped_windows += 1

            if int(w.real_len) < self.min_real_len:
                dropped_windows += 1
                self._log_debug(
                    "window_dropped",
                    request_id=request_id,
                    dst_ip=w.dst_ip,
                    real_len=int(w.real_len),
                    min_real_len=self.min_real_len,
                )
                if not self.send_dropped_windows:
                    continue

                events.append(
                    {
                        "dst_ip": w.dst_ip,
                        "real_len": int(w.real_len),
                        "decision": {"source": "DROP", "attack": False, "attack_types": [], "reason": "too_short"},
                        "meta": {
                            "seq_len": self.seq_len,
                            "stride": self.stride,
                            "ts": time.time(),
                            "request_id": request_id,
                            "schema_id": self.schema_id,
                            "config_sha256": self.config_sha256,
                            "numeric_cols_hash": self.numeric_cols_hash,
                            "model_version": self.model_version,
                        },
                    }
                )
                continue

            numeric, cat, mask, win_dbg = self._build_window_tensors(w)

            # ---- TCN ----
            logits = self.tcn_model(numeric, cat, mask)
            probs = torch.sigmoid(logits)[0]
            preds = probs >= 0.5

            k = min(3, self.num_classes)
            topv, topi = torch.topk(probs, k=k)
            topk = [{"idx": int(i.item()), "label": self.label_classes[int(i.item())], "prob": float(v.item())} for v, i in zip(topv, topi)]

            tcn_attack_classes = [self.label_classes[j] for j in range(1, self.num_classes) if bool(preds[j].item())]
            tcn_attack_any = len(tcn_attack_classes) > 0

            # ---- AE ----
            ae_used = False
            ae_score = None
            ae_attack = False

            # ✅ 설계 그대로: TCN이 benign이면 AE로 2차 검사
            if not tcn_attack_any:
                recon = self.ae_model(numeric, cat, mask)
                scores = compute_ae_scores(recon=recon, target=numeric, mask=mask, agg_mode=self.ae_agg_mode, topk=self.ae_topk)
                ae_score = float(scores[0].item())
                ae_used = True
                ae_attack = ae_score >= self.ae_threshold

            # ---- decision ----
            if tcn_attack_any:
                decision = {
                    "source": "TCN",
                    "attack": True,
                    "attack_types": tcn_attack_classes,
                    "confidence": {self.label_classes[j]: float(probs[j].item()) for j in range(1, self.num_classes) if bool(preds[j].item())},
                    "topk": topk,
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
                    "tcn_topk": topk,
                }
                ae_attack_windows += 1
            else:
                decision = {"source": "NONE", "attack": False, "attack_types": [], "topk": topk}

            # ---- debug meta (원인 추적용, 모델 입력에는 영향 없음) ----
            imputed_ratios = [float(r.get("_debug_imputed_ratio", 0.0)) for r in w.rows]
            miss_ns = [int(r.get("_debug_missing_n", 0)) for r in w.rows]
            imp_ns = [int(r.get("_debug_imputed_n", 0)) for r in w.rows]

            imp_ratio_mean = float(np.mean(imputed_ratios)) if imputed_ratios else 0.0
            miss_n_mean = float(np.mean(miss_ns)) if miss_ns else 0.0
            imp_n_mean = float(np.mean(imp_ns)) if imp_ns else 0.0

            pcap_sources = sorted({str(r.get("_debug_pcap")) for r in w.rows if r.get("_debug_pcap")})[:3]
            csv_sources = sorted({str(r.get("_debug_csv")) for r in w.rows if r.get("_debug_csv")})[:3]
            header_hash = sorted({str(r.get("_debug_header_hash")) for r in w.rows if r.get("_debug_header_hash")})[:3]

            meta = {
                "seq_len": self.seq_len,
                "stride": self.stride,
                "ae_used": ae_used,
                "ts": time.time(),
                "request_id": request_id,
                "schema_id": self.schema_id,
                "model_version": self.model_version,
                "config_sha256": self.config_sha256,
                "numeric_cols_hash": self.numeric_cols_hash,
            }

            if self.splunk_debug_meta:
                meta["debug"] = {
                    "pcap": pcap_sources,
                    "csv": csv_sources,
                    "header_hash": header_hash,
                    "win": win_dbg,
                    "imputer_ratio_mean": imp_ratio_mean,
                    "missing_n_mean": miss_n_mean,
                    "imputed_n_mean": imp_n_mean,
                }

            events.append({"dst_ip": w.dst_ip, "real_len": int(w.real_len), "decision": decision, "meta": meta})

            if wi < self.debug_windows:
                self._log_debug(
                    "window_debug",
                    request_id=request_id,
                    dst_ip=w.dst_ip,
                    real_len=int(w.real_len),
                    topk=topk,
                    decision_source=decision.get("source"),
                    attack=bool(decision.get("attack")),
                    attack_types=decision.get("attack_types"),
                    win=win_dbg,
                    imputer_ratio_mean=imp_ratio_mean,
                    missing_n_mean=miss_n_mean,
                    imputed_n_mean=imp_n_mean,
                    ae_score=ae_score,
                )

        if hec is not None and events:
            sent_events = hec.send_events(events)

        self._log_debug(
            "ingest_end",
            request_id=request_id,
            input_flows=len(flows),
            popped_windows=popped_windows,
            dropped_windows=dropped_windows,
            events=len(events),
            tcn_attack_windows=tcn_attack_windows,
            ae_attack_windows=ae_attack_windows,
            sent_events=sent_events,
        )

        return {
            "ok": True,
            "request_id": request_id,
            "input_flows": len(flows),
            "popped_windows": popped_windows,
            "dropped_windows": dropped_windows,
            "tcn_attack_windows": tcn_attack_windows,
            "ae_attack_windows": ae_attack_windows,
            "sent_events": sent_events,
        }

    def _build_window_tensors(self, w: WindowItem) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # ts_nonmono는 "정렬 전" 원본 순서에서 측정
        orig_ts = [r.get("Timestamp") for r in w.rows if r.get("Timestamp") is not None]
        ts_nonmono = 0.0
        if len(orig_ts) >= 2:
            bad = 0
            for i in range(1, len(orig_ts)):
                try:
                    if orig_ts[i] < orig_ts[i - 1]:
                        bad += 1
                except Exception:
                    pass
            ts_nonmono = bad / max(1, (len(orig_ts) - 1))

        rows = list(w.rows)
        real_len = int(w.real_len)
        L = self.seq_len

        if self.sort_by_timestamp:
            try:
                rows.sort(key=lambda r: r.get("Timestamp") if r.get("Timestamp") is not None else pd.Timestamp("1970-01-01"))
            except Exception:
                pass

        base = np.zeros((real_len, len(self.numeric_cols)), dtype=np.float32)
        for i, r in enumerate(rows):
            for j, c in enumerate(self.numeric_cols):
                try:
                    base[i, j] = float(r.get(c, 0.0))
                except Exception:
                    base[i, j] = 0.0

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
            try:
                t_min = min(ts_list)
                t_max = max(ts_list)
                window_duration = float(max((t_max - t_min).total_seconds(), 1e-6))
            except Exception:
                window_duration = 1e-6
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
        win_feat_real = np.repeat(win_feat[None, :], real_len, axis=0)
        numeric_real = np.concatenate([base, win_feat_real], axis=1)

        numeric = np.zeros((L, numeric_real.shape[1]), dtype=np.float32)
        numeric[:real_len] = numeric_real

        cat = np.zeros((L, 3), dtype=np.int64)
        for i, r in enumerate(rows):
            cat[i, 0] = int(r.get("sport_idx", 0))
            cat[i, 1] = int(r.get("dport_idx", 0))
            cat[i, 2] = int(r.get("proto_idx", 0))

        mask = np.zeros((L,), dtype=np.float32)
        mask[:real_len] = 1.0

        numeric_t = torch.tensor(numeric[None, :, :], dtype=torch.float32, device=self.device)
        cat_t = torch.tensor(cat[None, :, :], dtype=torch.int64, device=self.device)
        mask_t = torch.tensor(mask[None, :], dtype=torch.float32, device=self.device)

        try:
            var_mean = float(np.mean(np.var(base, axis=0)))
        except Exception:
            var_mean = 0.0

        win_dbg = {
            "unique_src": float(unique_src),
            "avg_flows_per_src": float(avg_flows_per_src),
            "src_entropy": float(src_ent_raw),
            "duration_s": float(window_duration),
            "flow_rate": float(flow_rate),
            "ts_nonmono_ratio": float(ts_nonmono),
            "base_var_mean": float(var_mean),
        }

        return numeric_t, cat_t, mask_t, win_dbg
