# src/engine/online_preprocess.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ---------------------------
# 컬럼 rename / drop 규칙
# ---------------------------
# master_raw(학습) 헤더 기준으로 맞춰주는 매핑
RENAME_MAP: Dict[str, str] = {
    # IP/Port (CICFlowMeter V3/V4 헤더 ↔ 학습 헤더)
    "Src IP": "Source IP",
    "Dst IP": "Destination IP",
    "Src Port": "Source Port",
    "Dst Port": "Destination Port",

    # --- CICFlowMeter 약어 버전 (자주 나오는 것들) ---
    # totals
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",

    # sometimes-cased / variants
    "Total Fwd Packet": "Total Fwd Packets",
    "Total Fwd Packet(s)": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Bwd Packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Fwd Packet(s)": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Total Length of Bwd Packet(s)": "Total Length of Bwd Packets",

    # bytes/packets per sec
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",

    # packet length min/max (약어/풀네임 혼재)
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Packet Length Min": "Min Packet Length",
    "Packet Length Max": "Max Packet Length",

    # avg segment size
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size",
    "Bwd Segment Size Avg": "Avg Bwd Segment Size",

    # header length
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",

    # init window bytes
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",

    # bulk features
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",

    # act/min seg
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",

    # flag count 약어
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",

    # CWR/CWE 표기 흔한 차이
    "CWR Flag Count": "CWE Flag Count",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _norm_key(k: Any) -> str:
    """BOM 제거 + 공백 정규화"""
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _is_bad_number(v: Any) -> bool:
    """None/NaN/Inf/빈문자/숫자변환실패는 전부 median으로 치환"""
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
    """
    전처리 진입 전에:
      - key 정규화(strip/BOM/공백)
      - rename
      - drop
      - Fwd Header Length.1 복제(학습 스키마 맞추기)
    """
    out: Dict[str, Any] = {}

    for k, v in row_raw.items():
        kk = _norm_key(k)
        if not kk:
            continue

        # drop
        if kk in DROP_KEYS:
            if kk == "Label" and (not drop_label):
                pass
            else:
                continue

        # rename
        kk = RENAME_MAP.get(kk, kk)
        out[kk] = v

    # 학습에 'Fwd Header Length.1'이 들어갔으면 실데이터에도 맞춰줘야 함
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
    학습(second_preprocess)과 최대한 동일하게:
      - rename/drop/복제(Fwd Header Length.1)
      - NaN/Inf/이상값 -> train median
      - Standard scaling (train mean/std)
      - sport/dport/proto index 생성
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        median = cfg.get("imputer", {}).get("median")
        if not isinstance(median, dict) or not median:
            raise RuntimeError(
                "preprocess_config.json에 imputer.median이 없습니다. "
                "second_preprocess에서 train median 저장 버전을 사용하세요."
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

            # 2) 필수 raw 필드
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

            # 5) numeric: bad -> median -> scale
            for c in num_cols:
                med = self._to_float(self.cfg.median.get(c, 0.0), 0.0)

                val = row.get(c, None)
                if _is_bad_number(val):
                    x = med
                else:
                    x = self._to_float(val, med)

                mu = self._to_float(self.cfg.mean.get(c, 0.0), 0.0)
                sd = self._to_float(self.cfg.std.get(c, 1.0), 1.0)
                if sd == 0:
                    sd = 1e-6

                row[c] = (x - mu) / sd

            out.append(row)

        return out
