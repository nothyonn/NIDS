# src/engine/online_preprocess.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------
# ★ 추가: 컬럼 rename / drop 규칙
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
    "Avg Fwd Segment Size": "Avg Fwd Segment Size",
    "Avg Bwd Segment Size": "Avg Bwd Segment Size",

    # init window bytes (V3 vs master_raw)
    "FWD Init Win Bytes": "Init_Win_bytes_forward",
    "Bwd Init Win Bytes": "Init_Win_bytes_backward",

    # CWR/CWE 표기 흔한 차이
    "CWR Flag Count": "CWE Flag Count",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _rename_and_clean(row_raw: Dict[str, Any], *, drop_label: bool) -> Dict[str, Any]:
    """전처리 진입 전에: key strip + rename + drop + Fwd Header Length.1 복제"""
    out: Dict[str, Any] = {}

    for k, v in row_raw.items():
        if k is None:
            continue
        kk = str(k).strip()

        # drop
        if kk in DROP_KEYS:
            if kk == "Label" and (not drop_label):
                pass
            else:
                continue

        # rename
        kk = RENAME_MAP.get(kk, kk)

        out[kk] = v

    # 학습에 'Fwd Header Length.1'이 들어갔으면, 실데이터에도 맞춰줘야 함
    # (없으면 median으로 대체되어 feature가 죽음)
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
    median: Dict[str, float]  # 반드시 있어야 함


class OnlinePreprocessor:
    """
    센서가 CICFlowMeter로 만든 flow(dict)를 받아서,
    학습 때 second_preprocess와 동일한 규칙으로:
      - NaN -> train median
      - Standard scaling (train mean/std)
      - sport/dport/proto index 생성
    까지 끝낸 row(dict)로 만든다.
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path).resolve()
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        median = cfg.get("imputer", {}).get("median")
        if not isinstance(median, dict) or not median:
            raise RuntimeError(
                "preprocess_config.json에 imputer.median이 없습니다. "
                "second_preprocess에서 train median을 저장하도록 넣은 버전으로 맞추세요."
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
            # ✅ (수정) strip만 하지 말고 rename/drop/복제까지
            row = _rename_and_clean(raw, drop_label=drop_label)

            # 2) 필수 raw 필드
            sport = row.get("Source Port", row.get("Src Port"))
            dport = row.get("Destination Port", row.get("Dst Port"))
            proto = row.get("Protocol")

            row["Source Port"] = self._to_int(sport, self.cfg.other_port_idx)
            row["Destination Port"] = self._to_int(dport, self.cfg.other_port_idx)
            row["Protocol"] = self._to_int(proto, self.cfg.proto_other_idx)

            # 3) Timestamp 파싱 (없으면 epoch)
            ts = row.get("Timestamp")
            ts = pd.to_datetime(str(ts).strip(), errors="coerce") if ts is not None else pd.NaT
            if pd.isna(ts):
                ts = pd.Timestamp("1970-01-01")
            row["Timestamp"] = ts

            # 4) port/proto idx
            row["sport_idx"] = self._map_port(row["Source Port"])
            row["dport_idx"] = self._map_port(row["Destination Port"])
            row["proto_idx"] = self._map_proto(row["Protocol"])

            # 5) numeric_cols를 median으로 채우고 scaling
            for c in num_cols:
                med = self._to_float(self.cfg.median.get(c, 0.0), 0.0)
                x = self._to_float(row.get(c), med)
                mu = self._to_float(self.cfg.mean.get(c, 0.0), 0.0)
                sd = self._to_float(self.cfg.std.get(c, 1.0), 1.0)
                if sd == 0:
                    sd = 1e-6
                row[c] = (x - mu) / sd

            out.append(row)

        return out
