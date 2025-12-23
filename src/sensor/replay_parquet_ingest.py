# src/sensor/replay_parquet_to_ingest.py
from __future__ import annotations

import argparse
import pandas as pd
import requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--ingest-url", required=True)  # http://MODEL_IP:8001/ingest
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=0)  # 0이면 전체
    ap.add_argument("--max-batches", type=int, default=10)
    ap.add_argument("--drop-label", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.drop_label and "Label" in df.columns:
        df = df.drop(columns=["Label"])

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    total = len(df)
    print("rows:", total)

    idx = 0
    b = 0
    while idx < total:
        chunk = df.iloc[idx: idx + args.batch]
        payload = {
            "flows": chunk.to_dict(orient="records"),
            "run_max_batches": args.max_batches,
            "drop_label": False,  # 이미 드랍했으면 false
        }
        r = requests.post(args.ingest_url, json=payload, timeout=30)
        print("POST", b, "status", r.status_code, r.text[:200])
        r.raise_for_status()

        idx += args.batch
        b += 1

if __name__ == "__main__":
    main()
