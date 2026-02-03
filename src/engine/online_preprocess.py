# src/engine/online_preprocess.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_RENAME_MAP: Dict[str, str] = {
    "Src IP": "Source IP",
    "Dst IP": "Destination IP",
    "Src Port": "Source Port",
    "Dst Port": "Destination Port",

    "Total Fwd Packet": "Total Fwd Packets",
    "Total Fwd Packet(s)": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Bwd Packets": "Total Backward Packets",

    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Fwd Packet(s)": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Total Length of Bwd Packet(s)": "Total Length of Bwd Packets",

    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",

    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",

    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",

    "CWR Flag Count": "CWE Flag Count",

    # ✅ Bulk (CICFlowMeter 변형들)
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Packets/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",

    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Packets/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _norm_key(k: Any) -> str:
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _canon(s: str) -> str:
    """
    키 비교를 위한 캐노니컬 형태.
    - lower
    - 알파넘/숫자 제외는 공백으로
    - 공백 collapse
    """
    s = _norm_key(s).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s


def _coerce_nan_like(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("", "nan", "na", "none", "null"):
            return None
        if s in ("infinity", "+infinity", "-infinity", "inf", "+inf", "-inf"):
            return None

    try:
        fv = float(v)
        if not math.isfinite(fv):
            return None
    except Exception:
        return None

    return v


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
            raise RuntimeError("preprocess_config.json에 imputer.median이 없습니다.")

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

        # ✅ “들어오는 키가 뭐든” numeric_cols의 정확한 키로 붙잡기 위한 dict
        self._canon_to_exact: Dict[str, str] = {_canon(c): c for c in self.cfg.numeric_cols}

        # ✅ 추가로 자주 쓰는 raw 필드도 캐노니컬 매칭 (혹시 센서가 이상하게 보내도 살림)
        for extra in ["Source IP", "Destination IP", "Timestamp", "Protocol", "Source Port", "Destination Port"]:
            self._canon_to_exact.setdefault(_canon(extra), extra)

        # rename_map 확정 (BASE + config 기반으로 act/minseg도 안전하게 흡수)
        self.rename_map = dict(BASE_RENAME_MAP)

        # config에 실제 들어있는 이름으로 유도 (프로젝트마다 표기 다를 수 있음)
        nc = set(self.cfg.numeric_cols)

        def pick(*cands: str) -> str | None:
            for c in cands:
                if c in nc:
                    return c
            return None

        act_target = pick("act_data_pkt_fwd", "Act data pkt fwd", "act data pkt fwd", "Fwd Act Data Pkts")
        minseg_target = pick("min_seg_size_forward", "Min Seg Size Forward", "min seg size forward", "Fwd Seg Size Min")

        if act_target is not None:
            self.rename_map["Fwd Act Data Pkts"] = act_target
        if minseg_target is not None:
            self.rename_map["Fwd Seg Size Min"] = minseg_target

        # ✅ scaler/median 키 커버리지 체크
        nc_set = set(self.cfg.numeric_cols)
        miss_mean = nc_set - set(self.cfg.mean.keys())
        miss_std = nc_set - set(self.cfg.std.keys())
        miss_med = nc_set - set(self.cfg.median.keys())
        if miss_mean or miss_std or miss_med:
            raise RuntimeError(
                "preprocess_config.json scaler/median 키가 numeric_cols와 불일치합니다.\n"
                f"- missing mean: {sorted(list(miss_mean))[:10]} (total {len(miss_mean)})\n"
                f"- missing std : {sorted(list(miss_std))[:10]} (total {len(miss_std)})\n"
                f"- missing med : {sorted(list(miss_med))[:10]} (total {len(miss_med)})\n"
            )

    def rename_and_clean_only(self, row_raw: Dict[str, Any], *, drop_label: bool) -> Dict[str, Any]:
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

            # 1) base rename
            kk = self.rename_map.get(kk, kk)

            # 2) canonical match → exact key
            ck = _canon(kk)
            if ck in self._canon_to_exact:
                kk = self._canon_to_exact[ck]

            out[kk] = v

        if "Fwd Header Length.1" not in out and "Fwd Header Length" in out:
            out["Fwd Header Length.1"] = out["Fwd Header Length"]

        return out

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

    def transform(
        self,
        flows: List[Dict[str, Any]],
        *,
        drop_label: bool = True,
        attach_debug: bool = True,
        debug_top_missing: int = 20,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        num_cols = self.cfg.numeric_cols

        for raw in flows:
            row = self.rename_and_clean_only(raw, drop_label=drop_label)

            sport = row.get("Source Port", row.get("Src Port"))
            dport = row.get("Destination Port", row.get("Dst Port"))
            proto = row.get("Protocol")

            row["Source Port"] = self._to_int(sport, self.cfg.other_port_idx)
            row["Destination Port"] = self._to_int(dport, self.cfg.other_port_idx)
            row["Protocol"] = self._to_int(proto, self.cfg.proto_other_idx)

            ts = row.get("Timestamp")
            ts = pd.to_datetime(str(ts).strip(), errors="coerce") if ts is not None else pd.NaT
            if pd.isna(ts):
                ts = pd.Timestamp("1970-01-01")
            row["Timestamp"] = ts

            row["sport_idx"] = self._map_port(row["Source Port"])
            row["dport_idx"] = self._map_port(row["Destination Port"])
            row["proto_idx"] = self._map_proto(row["Protocol"])

            imputed_cols: List[str] = []
            missing_cols: List[str] = []

            for c in num_cols:
                if c not in row:
                    missing_cols.append(c)

                med = self._to_float(self.cfg.median.get(c, 0.0), 0.0)
                mu = self._to_float(self.cfg.mean.get(c, 0.0), 0.0)
                sd = self._to_float(self.cfg.std.get(c, 1.0), 1.0)
                if sd == 0:
                    sd = 1e-6

                v = _coerce_nan_like(row.get(c, None))
                if v is None:
                    x = med
                    imputed_cols.append(c)
                else:
                    x = self._to_float(v, med)
                    if x == med and v != med:
                        imputed_cols.append(c)

                row[c] = (x - mu) / sd

            if attach_debug:
                row["_debug_missing_numeric_cols"] = missing_cols[:debug_top_missing]
                row["_debug_missing_n"] = int(len(missing_cols))
                row["_debug_imputed_n"] = int(len(imputed_cols))
                row["_debug_imputed_ratio"] = float(len(imputed_cols) / max(1, len(num_cols)))

            out.append(row)

        return out
