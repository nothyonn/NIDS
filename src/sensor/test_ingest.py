# src/sensor/test_ingest.py
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_ts_str() -> str:
    # 모델 코드가 문자열로 strip/처리하니까 문자열로 넣기
    # (형식 엄격하면 CICFlowMeter 형식으로 바꾸면 됨)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_flow(cfg: dict, *, sport: int, dport: int, proto: int) -> Dict[str, Any]:
    numeric_cols: List[str] = cfg["numeric_cols"]
    med: Dict[str, Any] = cfg.get("imputer", {}).get("median", {})

    flow: Dict[str, Any] = {}

    # ===== (A) FlowWindowDataset이 기대하는 "기본 필드"들 =====
    flow["Timestamp"] = now_ts_str()
    flow["Source IP"] = "192.168.8.131"
    flow["Destination IP"] = "192.168.8.130"

    # CICFlowMeter 스타일 raw 필드
    flow["Source Port"] = int(sport)
    flow["Destination Port"] = int(dport)
    flow["Protocol"] = int(proto)

    # 있으면 도움이 되는 메타 (없어도 되지만, 일부 코드가 기대하면 안전)
    flow["source_file"] = "sensor_ingest_test"

    # ===== (B) numeric feature들: median으로 채워서 NaN 방지 =====
    for c in numeric_cols:
        v = med.get(c, 0.0)
        flow[c] = float(v) if v is not None else 0.0

    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-url", default="http://192.168.8.130:8001")
    ap.add_argument("--config", default="processed/preprocess_config.json")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--run-max-batches", type=int, default=10)
    ap.add_argument("--drop-label", action="store_true")
    ap.add_argument("--sport", type=int, default=80)
    ap.add_argument("--dport", type=int, default=12345)
    ap.add_argument("--proto", type=int, default=6)  # TCP=6, UDP=17
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # NIDS/
    config_path = (root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_config(config_path)

    flows: List[Dict[str, Any]] = []
    for i in range(args.n):
        flows.append(build_flow(cfg, sport=args.sport, dport=args.dport + i, proto=args.proto))

    payload = {
        "flows": flows,
        "run_max_batches": args.run_max_batches,
        "drop_label": bool(args.drop_label),
    }

    url = f"{args.model_url.rstrip('/')}/ingest"
    r = requests.post(url, json=payload, timeout=120)
    print("POST", url)
    print("status:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    random.seed(42)
    main()
