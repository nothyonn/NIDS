# src/sensor/test_csv_ingest.py
import csv, json
import requests
from pathlib import Path

CSV_PATH = Path("/data/flows/test2.pcap_ISCX.csv")   # <- 네 파일명에 맞게
MODEL_URL = "http://192.168.0.12:8001/ingest"       # <- 데탑 모델서버

LIMIT = 500          # 일단 500개만 보내서 확인
BATCH = 200          # 200개씩 나눠서 POST

def read_rows(p, limit):
    rows = []
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if limit and i >= limit:
                break
            # 키/값 공백 정리 (CICFlowMeter CSV 헤더가 공백 포함이라)
            row2 = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k,v in row.items()}
            rows.append(row2)
    return rows

rows = read_rows(CSV_PATH, LIMIT)
print("loaded rows:", len(rows))

sent = 0
for i in range(0, len(rows), BATCH):
    chunk = rows[i:i+BATCH]
    payload = {
        "flows": chunk,
        "run_max_batches": 999999,  # 모델이 알아서 윈도잉/추론 돌림
        "drop_label": True          # live CSV엔 라벨 없을 수 있으니 드랍
    }
    resp = requests.post(MODEL_URL, json=payload, timeout=180)
    print("POST", i, "~", i+len(chunk), "status", resp.status_code, "body", resp.text[:200])
    sent += len(chunk)

print("done, sent:", sent)
