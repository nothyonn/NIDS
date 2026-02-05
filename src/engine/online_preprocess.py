# src/engine/online_preprocess.py
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------
# 컬럼 rename / drop 규칙
# ---------------------------
BASE_RENAME_MAP: Dict[str, str] = {
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

    # ✅ CICFlowMeter V3 Bulk 이름 → 학습 스키마 이름(가장 흔한 매칭)
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk",
    "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
}

DROP_KEYS = {"Flow ID", "Label", "source_file"}


def _norm_key(k: Any) -> str:
    if k is None:
        return ""
    s = str(k).replace("\ufeff", "").strip()
    s = " ".join(s.split())
    return s


def _coerce_nan_like(v: Any) -> Optional[float]:
    """
    ✅ 학습 전처리(master_raw/second_preprocess) 의도와 동일하게:
    - "", "NaN", "Infinity", "inf", "-inf" 등은 NaN 취급 -> None 반환
    - 숫자 변환 실패도 None
    - 유한한 숫자면 float로 반환
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

    try:
        fv = float(v)
        if not math.isfinite(fv):
            return None
        return fv
    except Exception:
        return None


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

    + (배포 안정화 패치)
    - 학습에서 사실상 상수(std가 극소)였던 컬럼 / Bulk 계열 컬럼은 실시간 튐값이 들어오면 z-score가 폭발함
      => 해당 컬럼은 median으로 강제 고정(학습 분포 유지)
    - scaling 결과 z-score clip으로 폭발값 방지
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

        # ✅ scaler/median 키가 numeric_cols를 커버하는지 검증
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
            )

        # ✅ config 기준으로 프로젝트마다 이름이 달라지는 컬럼(대표 2개)을 안전하게 매칭
        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                if c in nc:
                    return c
            return None

        # act_data / min_seg 는 학습 스키마 기준 우선
        self._act_data_target = pick("act_data_pkt_fwd", "Act data pkt fwd", "act data pkt fwd", "Fwd Act Data Pkts")
        self._min_seg_target = pick("min_seg_size_forward", "Min Seg Size Forward", "Fwd Seg Size Min")

        # rename_map 확정
        self.rename_map: Dict[str, str] = dict(BASE_RENAME_MAP)

        # CICFlowMeter V3 원본 이름들을 config가 기대하는 이름으로 맞추기
        if self._act_data_target is not None:
            self.rename_map["Fwd Act Data Pkts"] = self._act_data_target

        if self._min_seg_target is not None:
            self.rename_map["Fwd Seg Size Min"] = self._min_seg_target

        # ---------------------------
        # 안정화 패치 설정
        # ---------------------------
        # z-score clip (폭발 방지)
        self.z_clip = 10.0  # 하드코딩

        # 학습에서 상수였던 컬럼 / Bulk 계열은 median으로 강제 고정
        def _sf(x: Any, default: float) -> float:
            try:
                fv = float(x)
                if not math.isfinite(fv):
                    return default
                return fv
            except Exception:
                return default

        # Bulk 계열은 우선 강제로 포함 (학습/실시간 분포 mismatch로 자주 터짐)
        bulk_like = {
            "Fwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
        }

        self.force_zero_cols: set[str] = set()
        self.force_zero_cols.update({c for c in bulk_like if c in nc})

        # std가 극소인(사실상 상수) 컬럼도 강제 고정
        # (학습 때 거의 변동이 없었던 피처는 실시간에서 값이 튀면 (x-mu)/sd가 폭발)
        for c in self.cfg.numeric_cols:
            sd = _sf(self.cfg.std.get(c, 1.0), 1.0)
            if sd <= 1e-6 + 1e-12:
                self.force_zero_cols.add(c)

        # 디버그/재현성
        self.force_zero_cols_sorted = sorted(self.force_zero_cols)

    def rename_and_clean_only(self, row_raw: Dict[str, Any], *, drop_label: bool) -> Dict[str, Any]:
        """
        ✅ rename/drop/복제만 수행(스키마 확인용)
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

            kk = self.rename_map.get(kk, kk)
            out[kk] = v

        # 학습 스키마 맞추기
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
            fv = float(v)
            if not math.isfinite(fv):
                return default
            return fv
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
            row = self.rename_and_clean_only(raw, drop_label=drop_label)

            # 2) 필수 raw 필드(포트/프로토콜)
            sport = row.get("Source Port", row.get("Src Port"))
            dport = row.get("Destination Port", row.get("Dst Port"))
            proto = row.get("Protocol")

            row["Source Port"] = self._to_int(sport, self.cfg.other_port_idx)
            row["Destination Port"] = self._to_int(dport, self.cfg.other_port_idx)
            row["Protocol"] = self._to_int(proto, self.cfg.proto_other_idx)

            # 3) Timestamp 파싱 강화:
            #    1) CICFlowMeter 대표 포맷 우선
            #    2) 실패 시 일반 파싱
            #    3) 그래도 실패면 센서 수집시각(_debug_sensor_ts)로 대체
            #    4) 최후에 1970
            ts_raw = row.get("Timestamp")
            ts = pd.NaT

            if ts_raw is not None:
                s = str(ts_raw).strip()
                if s:
                    # CICFlowMeter 예: "12/01/2026 05:59:33 AM"
                    ts = pd.to_datetime(s, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
                    if pd.isna(ts):
                        ts = pd.to_datetime(s, errors="coerce")

            if pd.isna(ts):
                sensor_ts = raw.get("_debug_sensor_ts", None)
                if sensor_ts is None:
                    sensor_ts = row.get("_debug_sensor_ts", None)
                if sensor_ts is not None:
                    try:
                        ts = pd.to_datetime(float(sensor_ts), unit="s", errors="coerce")
                    except Exception:
                        ts = pd.NaT

            if pd.isna(ts):
                ts = pd.Timestamp("1970-01-01")

            row["Timestamp"] = ts

            # 4) port/proto idx
            row["sport_idx"] = self._map_port(row["Source Port"])
            row["dport_idx"] = self._map_port(row["Destination Port"])
            row["proto_idx"] = self._map_proto(row["Protocol"])

            # 5) numeric_cols: (Infinity/NaN/inf 처리) -> (force-zero) -> median -> scaling -> clip
            imputed_cols: List[str] = []
            missing_cols: List[str] = []
            force_zero_applied: List[str] = []

            for c in num_cols:
                if c not in row:
                    missing_cols.append(c)

                med = self._to_float(self.cfg.median.get(c, 0.0), 0.0)
                mu = self._to_float(self.cfg.mean.get(c, 0.0), 0.0)
                sd = self._to_float(self.cfg.std.get(c, 1.0), 1.0)
                if sd == 0:
                    sd = 1e-6

                # ✅ 핵심: 학습에서 상수/민감 컬럼은 실시간 값 무시하고 median 고정
                if c in self.force_zero_cols:
                    x = med
                    imputed_cols.append(c)
                    force_zero_applied.append(c)
                else:
                    vraw = row.get(c, None)
                    fv = _coerce_nan_like(vraw)
                    if fv is None:
                        x = med
                        imputed_cols.append(c)
                    else:
                        x = fv

                z = (x - mu) / sd

                # ✅ 폭발값 방지 clip
                if z > self.z_clip:
                    z = self.z_clip
                elif z < -self.z_clip:
                    z = -self.z_clip

                row[c] = float(z)

            if attach_debug:
                row["_debug_missing_numeric_cols"] = missing_cols[:debug_top_missing]
                row["_debug_missing_n"] = int(len(missing_cols))
                row["_debug_imputed_n"] = int(len(imputed_cols))
                row["_debug_imputed_ratio"] = float(len(imputed_cols) / max(1, len(num_cols)))

                # force-zero 디버그(폭발 원인 추적용)
                row["_debug_force_zero_n"] = int(len(force_zero_applied))
                row["_debug_force_zero_cols"] = force_zero_applied[:debug_top_missing]

            out.append(row)

        return out
