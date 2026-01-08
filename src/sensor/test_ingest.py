# src/sensor/test_ingest.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta

import requests


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_port(cfg: dict, port: int) -> int:
    port_map = {int(k): int(v) for k, v in cfg["port_idx_map"].items()}
    other = int(cfg["other_port_idx"])
    return port_map.get(int(port), other)


def map_proto(cfg: dict, proto: int) -> int:
    proto_map = {int(k): int(v) for k, v in cfg["proto_idx_map"].items()}
    other = int(cfg["proto_other_idx"])
    return proto_map.get(int(proto), other)


def build_scaled_numeric(cfg: dict) -> dict:
    """
    flows_*_scaled.parquet이 '스케일된 numeric'을 들고 있으니까,
    median(원본 스케일) 값을 (median-mean)/std 로 스케일해서 넣어줌.
    """
    numeric_cols = cfg["numeric_cols"]
    med = cfg.get("imputer", {}).get("median", {})
    mean = cfg.get("scaler", {}).get("mean", {})
    std = cfg.get("scaler", {}).get("std", {})

    out = {}
    for c in numeric_cols:
        m = float(med.get(c, 0.0))
        mu = float(mean.get(c, 0.0))
        sd = float(std.get(c, 1.0)) if float(std.get(c, 1.0)) != 0.0 else 1e-6
        out[c] = (m - mu) / sd
    return out


def build_one_flow(
    cfg: dict,
    *,
    ts: str,
    src_ip: str,
    dst_ip: str,
    sport: int,
    dport: int,
    proto: int,
    label: str = "BENIGN",
) -> dict:
    flow = {}

    # ---- raw fields (dataset_window가 필요로 하는 애들) ----
    flow["Timestamp"] = ts
    flow["Source IP"] = src_ip
    flow["Destination IP"] = dst_ip
    flow["Source Port"] = int(sport)
    flow["Destination Port"] = int(dport)
    flow["Protocol"] = int(proto)
    flow["source_file"] = "live_ingest"

    # ---- categorical indices (지금 에러의 핵심) ----
    flow["sport_idx"] = map_port(cfg, sport)
    flow["dport_idx"] = map_port(cfg, dport)
    flow["proto_idx"] = map_proto(cfg, proto)

    # ---- label (없으면 y 생성에서 터질 가능성 높음) ----
    flow["Label"] = label

    # ---- scaled numeric ----
    flow.update(build_scaled_numeric(cfg))

    return flow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-url", default="http://192.168.0.10:8001")
    ap.add_argument("--config", default="processed/preprocess_config.json")
    ap.add_argument("--n", type=int, default=200, help="전송할 flow 개수")
    ap.add_argument("--run-max-batches", type=int, default=10)
    ap.add_argument("--drop-label", action="store_true", help="서버에서 Label 드랍(권장X: dataset이 Label 필요할 수 있음)")
    ap.add_argument("--sport", type=int, default=80)
    ap.add_argument("--dport", type=int, default=12345)
    ap.add_argument("--proto", type=int, default=6)  # TCP=6, UDP=17
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]  # NIDS/
    cfg = load_config((root / args.config).resolve())

    base = datetime.now()
    flows = []
    for i in range(args.n):
        ts = (base + timedelta(milliseconds=10 * i)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        flows.append(
            build_one_flow(
                cfg,
                ts=ts,
                src_ip="10.0.0.1",
                dst_ip="10.0.0.2",
                sport=args.sport,
                dport=args.dport + i,  # 살짝만 변화
                proto=args.proto,
                label="BENIGN",
            )
        )

    payload = {
        "flows": flows,
        "run_max_batches": args.run_max_batches,
        "drop_label": bool(args.drop_label),
    }

    url = f"{args.model_url.rstrip('/')}/ingest"
    r = requests.post(url, json=payload, timeout=180)
    print("POST", url)
    print("status:", r.status_code)
    print(r.text)


if __name__ == "__main__":
    main()
