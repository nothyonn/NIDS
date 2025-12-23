# scripts/test_ingest.py
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import requests


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_flow_from_config(cfg: dict, *, sport: int, dport: int, proto: int) -> dict:
    """
    모델이 학습에서 쓰던 컬럼 스키마에 맞춰:
    - numeric_cols는 config.imputer.median 값으로 전부 채움 (NaN 방지)
    - Source/Destination Port, Protocol도 넣어줌
    """
    numeric_cols = cfg["numeric_cols"]
    med = cfg.get("imputer", {}).get("median", {})

    flow = {}

    # 1) 필수 categorical/raw fields (FlowWindowDataset에서 매핑할 가능성이 큼)
    flow["Source Port"] = int(sport)
    flow["Destination Port"] = int(dport)
    flow["Protocol"] = int(proto)

    # 2) numeric feature 전부 채우기 (기본은 train median)
    for c in numeric_cols:
        v = med.get(c, 0.0)
        # json 직렬화 안전하게 float로
        flow[c] = float(v) if v is not None else 0.0

    # 3) 테스트용으로 약간 흔들고 싶으면 일부 값 랜덤 노이즈 (원하면 주석 해제)
    # flow["Flow Duration"] *= random.uniform(0.8, 1.2)
    # flow["Flow Bytes/s"] *= random.uniform(0.8, 1.2)

    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-url", default="http://192.168.8.130:8001", help="예: http://192.168.8.130:8001")
    ap.add_argument("--config", default="processed/preprocess_config.json", help="NIDS 기준 상대경로")
    ap.add_argument("--n", type=int, default=5, help="전송할 flow 개수")
    ap.add_argument("--run-max-batches", type=int, default=5, help="모델에서 /ingest 실행 시 max_batches")
    ap.add_argument("--drop-label", action="store_true", help="payload에 Label 있어도 제거")
    ap.add_argument("--sport", type=int, default=80)
    ap.add_argument("--dport", type=int, default=12345)
    ap.add_argument("--proto", type=int, default=6)  # TCP=6, UDP=17
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # NIDS/
    config_path = (root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    cfg = load_config(config_path)

    flows = []
    for i in range(args.n):
        # 포트/프로토콜을 살짝 바꿔서 여러 케이스 보내고 싶으면 여기서 변형
        sport = args.sport
        dport = args.dport + i
        proto = args.proto
        flows.append(build_flow_from_config(cfg, sport=sport, dport=dport, proto=proto))

    payload = {
        "flows": flows,
        "run_max_batches": args.run_max_batches,
        "drop_label": bool(args.drop_label),
    }

    url = f"{args.model_url.rstrip('/')}/ingest"
    r = requests.post(url, json=payload, timeout=60)
    print("POST", url)
    print("status:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    random.seed(42)
    main()
