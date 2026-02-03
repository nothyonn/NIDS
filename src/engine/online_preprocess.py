# src/engine/online_preprocess.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ---------------------------
# 컬럼 rename / drop 규칙
# ---------------------------
RENAME_MAP: Dict[str, str] = {
    # IP/Port (CICFlowMeter V3 헤더 ↔ 학습 헤더)
    "Src IP": "Source IP",
    "Dst IP": "Destination IP",
    "Src Port": "Source Port",
    "Dst Port": "Destination Port",

    # packet count / length (자주 틀어지는 것들)
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Fwd Packet(s)": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Bwd Packets": "Total Backward Packets",

    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Fwd Packet(s)": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Total Length of Bwd Packet(s)": "Total Length of Bwd Packets",

    # packet length min/max (V3 vs master_raw)
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",

    # avg segment size (V3 vs master_raw)
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",

    # init window bytes (V3 vs master_raw)
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",

    # CWR/CWE 표기 흔한 차이
    "CWR Flag Count": "CWE Flag Count",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _norm_key(k: Any) -> str:
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _coerce_nan_like(v: Any) -> Any:
    """
    master_raw.py에서 하던 것과 동일한 의도:
    - object 컬럼에 들어오는 "Infinity"/"NaN" 같은 문자열을 NaN으로
    - 숫자형 inf/-inf도 NaN으로
    """
    if v is None:
        return None

    # pandas NaN
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

    # 숫자로 변환 가능한데 inf면 None
    try:
        fv = float(v)
        if not math.isfinite(fv):
            return None
    except Exception:
        # 숫자 변환 실패 -> None 처리(학습 전처리에서 to_numeric(errors="coerce")와 동치)
        return None

    return v


def rename_and_clean_only(row_raw: Dict[str, Any], *, drop_label: bool) -> Dict[str, Any]:
    """
    ✅ 여기 단계는 "스키마/rename 문제를 잡기 위한 단계"라서
    numeric_cols 강제 생성/스케일링/median 대체 같은 걸 절대 하지 않음.
    """
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

    # 학습 스키마 맞추기
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
    """
    학습 파이프라인과 일치:
    - master_raw: 문자열 Infinity/NaN/inf 처리 + to_numeric(errors="coerce") 의도
    - second_preprocess: NaN -> median, scaling(mean/std)
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        median = cfg.get("imputer", {}).get("median")
        if not isinstance(median, dict) or not median:
            raise RuntimeError(
                "preprocess_config.json에 imputer.median이 없습니다. "
                "second_preprocess에서 train median 저장한 config로 맞추세요."
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

        # ✅ 안전장치: scaler/median 키가 numeric_cols를 제대로 커버하는지 즉시 검증
        nc = set(self.cfg.numeric_cols)
        miss_mean = sorted(list(nc - set(self.cfg.mean.keys())))
        miss_std = sorted(list(nc - set(self.cfg.std.keys())))
        miss_med = sorted(list(nc - set(self.cfg.median.keys())))
        if miss_mean or miss_std or miss_med:
            raise RuntimeError(
                "preprocess_config.json scaler/median 키가 numeric_cols와 불일치합니다.\n"
                f"- missing mean: {miss_mean[:10]} (total {len(miss_mean)})\n"
                f"- missing std : {miss_std[:10]} (total {len(miss_std)})\n"
                f"- missing med : {miss_med[:10]} (total {len(miss_med)})\n"
                "=> 이 상태면 online scaling이 학습과 달라져서 AE가 전부 anomaly로 터질 수 있습니다."
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
            # 1) rename/drop/복제(스키마 단계)
            row = rename_and_clean_only(raw, drop_label=drop_label)

            # 2) 필수 raw 필드(포트/프로토콜)
            sport = row.get("Source Port", row.get("Src Port"))
            dport = row.get("Destination Port", row.get("Dst Port"))
            proto = row.get("Protocol")

            row["Source Port"] = self._to_int(sport, self.cfg.other_port_idx)
            row["Destination Port"] = self._to_int(dport, self.cfg.other_port_idx)
            row["Protocol"] = self._to_int(proto, self.cfg.proto_other_idx)

            # 3) Timestamp 파싱
            ts = row.get("Timestamp")
            ts = pd.to_datetime(str(ts).strip(), errors="coerce") if ts is not None else pd.NaT
            if pd.isna(ts):
                ts = pd.Timestamp("1970-01-01")
            row["Timestamp"] = ts

            # 4) port/proto idx
            row["sport_idx"] = self._map_port(row["Source Port"])
            row["dport_idx"] = self._map_port(row["Destination Port"])
            row["proto_idx"] = self._map_proto(row["Protocol"])

            # 5) numeric_cols: master_raw(to_numeric/coerce) + second_preprocess(fillna->median) + scaling
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
                    # 변환 실패로 default(med) 들어가면 impute로 기록
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
