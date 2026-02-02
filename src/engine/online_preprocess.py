from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


RENAME_MAP: Dict[str, str] = {
    # IP/Port
    "Src IP": "Source IP",
    "Dst IP": "Destination IP",
    "Src Port": "Source Port",
    "Dst Port": "Destination Port",

    # packet count / length
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Fwd Packet(s)": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Bwd Packets": "Total Backward Packets",

    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Fwd Packet(s)": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Total Length of Bwd Packet(s)": "Total Length of Bwd Packets",

    # packet length min/max
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",

    # avg segment size
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",

    # init window bytes
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",

    # CWR/CWE
    "CWR Flag Count": "CWE Flag Count",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _norm_key(k: Any) -> str:
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _is_bad_number(v: Any) -> bool:
    if v is None:
        return True

    try:
        if pd.isna(v):
            return True
    except Exception:
        pass

    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "nan", "na", "none", "null"):
            return True
        if s in ("inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"):
            return True

    try:
        fv = float(v)
        if not math.isfinite(fv):
            return True
    except Exception:
        return True

    return False


def _rename_and_clean(row_raw: Dict[str, Any], *, drop_label: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for k, v in row_raw.items():
        kk = _norm_key(k)
        if not kk:
            continue

        if kk in DROP_KEYS:
            if kk == "Label" and (not drop_label):
                pass
            else:
                continue

        kk = RENAME_MAP.get(kk, kk)
        out[kk] = v

    # 학습 스키마 맞춤 (있어야 하는 경우)
    if "Fwd Header Length.1" not in out and "Fwd Header Length" in out:
        out["Fwd Header Length.1"] = out["Fwd Header Length"]

    return out


@dataclass
class OnlinePreprocessConfig:
    numeric_cols: List[str]
    port_idx_map: Dict[str, int]
    other_port_idx: int
    proto_idx_map: Dict[str, int]
    proto_other_idx: int
    mean: Dict[str, float]
    std: Dict[str, float]
    median: Dict[str, float]


class OnlinePreprocessor:
    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        median = cfg.get("imputer", {}).get("median")
        if not isinstance(median, dict) or not median:
            raise RuntimeError(
                "preprocess_config.json에 imputer.median이 없습니다. "
                "second_preprocess에서 train median 저장 버전으로 맞추세요."
            )

        self.cfg = OnlinePreprocessConfig(
            numeric_cols=cfg["numeric_cols"],
            port_idx_map=cfg["port_idx_map"],
            other_port_idx=int(cfg["other_port_idx"]),
            proto_idx_map=cfg["proto_idx_map"],
            proto_other_idx=int(cfg["proto_other_idx"]),
            mean=cfg["scaler"]["mean"],
            std=cfg["scaler"]["std"],
            median=median,
        )

    @staticmethod
    def _to_int(v: Any, default: int) -> int:
        try:
            return int(float(v))
        except Exception:
            return default

    @staticmethod
    def _to_float(v: Any, default: float) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _map_port(self, p: Any) -> int:
        p_int = self._to_int(p, self.cfg.other_port_idx)
        return int(self.cfg.port_idx_map.get(str(p_int), self.cfg.other_port_idx))

    def _map_proto(self, v: Any) -> int:
        v_int = self._to_int(v, self.cfg.proto_other_idx)
        return int(self.cfg.proto_idx_map.get(str(v_int), self.cfg.proto_other_idx))

    def transform(self, flows: List[Dict[str, Any]], *, drop_label: bool = True) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        num_cols = self.cfg.numeric_cols

        for raw in flows:
            row = _rename_and_clean(raw, drop_label=drop_label)

            # ports/proto raw
            sport = row.get("Source Port", row.get("Src Port"))
            dport = row.get("Destination Port", row.get("Dst Port"))
            proto = row.get("Protocol")

            row["Source Port"] = self._to_int(sport, self.cfg.other_port_idx)
            row["Destination Port"] = self._to_int(dport, self.cfg.other_port_idx)
            row["Protocol"] = self._to_int(proto, self.cfg.proto_other_idx)

            # Timestamp
            ts = row.get("Timestamp")
            ts = pd.to_datetime(str(ts).strip(), errors="coerce") if ts is not None else pd.NaT
            if pd.isna(ts):
                ts = pd.Timestamp("1970-01-01")
            row["Timestamp"] = ts

            # categorical indices (이건 스케일링 전에 계산)
            row["sport_idx"] = self._map_port(row["Source Port"])
            row["dport_idx"] = self._map_port(row["Destination Port"])
            row["proto_idx"] = self._map_proto(row["Protocol"])

            # numeric cols: bad/missing -> median -> z-score
            imputed = 0
            for c in num_cols:
                med = self._to_float(self.cfg.median.get(c, 0.0), 0.0)

                val = row.get(c, None)
                if _is_bad_number(val):
                    x = med
                    imputed += 1
                else:
                    x = self._to_float(val, med)

                mu = self._to_float(self.cfg.mean.get(c, 0.0), 0.0)
                sd = self._to_float(self.cfg.std.get(c, 1.0), 1.0)
                if sd == 0:
                    sd = 1e-6

                row[c] = (x - mu) / sd

            # ★ 디버그: 이 row에서 median 대체가 얼마나 일어났는지
            row["_debug_imputed_n"] = int(imputed)
            row["_debug_imputed_ratio"] = float(imputed / max(1, len(num_cols)))

            out.append(row)

        return out
