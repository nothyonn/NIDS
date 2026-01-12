# src/sensor/pipeline.py
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set

import requests


def stable_file(path: Path, wait_sec: float = 1.0) -> bool:
    """파일 크기가 잠깐 동안 변하지 않으면 '닫혔다'고 간주."""
    try:
        s1 = path.stat().st_size
        time.sleep(wait_sec)
        s2 = path.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False


def run_cicflowmeter(jar: Path, pcap: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["java", "-jar", str(jar), "-f", str(pcap), "-c", str(out_csv)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_csv_rows(csv_path: Path) -> List[Dict]:
    """cicflowmeter csv를 row dict 리스트로 읽음(헤더 기반)."""
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 컬럼명/값 공백 정리
            rr = { (k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items() }
            rows.append(rr)
    return rows


def post_ingest(model_url: str, flows: List[Dict], timeout_sec: int = 180) -> requests.Response:
    url = f"{model_url.rstrip('/')}/ingest"
    payload = {
        "flows": flows,
        "run_max_batches": 999999,  # 모델쪽에서 내부적으로 윈도잉/추론하도록(필요하면 줄여도 됨)
        "drop_label": True,         # live flow에 Label 없을 확률 높으니 드랍
    }
    return requests.post(url, json=payload, timeout=timeout_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", default="/var/nids/pcap")
    ap.add_argument("--csv-dir", default="/var/nids/flowcsv")
    ap.add_argument("--cic-jar", required=True, help="예: /opt/cicflowmeter/CICFlowMeter.jar")
    ap.add_argument("--model-url", required=True, help="예: http://192.168.8.130:8001")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--batch-rows", type=int, default=200)
    ap.add_argument("--pcap-glob", default="web_*.pcap")
    args = ap.parse_args()

    pcap_dir = Path(args.pcap_dir)
    csv_dir = Path(args.csv_dir)
    jar = Path(args.cic_jar)

    seen: Set[str] = set()
    buffer: List[Dict] = []

    print("[sensor pipeline] start")
    print("  pcap_dir :", pcap_dir)
    print("  csv_dir  :", csv_dir)
    print("  jar      :", jar)
    print("  model_url:", args.model_url)

    while True:
        for pcap in sorted(pcap_dir.glob(args.pcap_glob)):
            key = pcap.name
            if key in seen:
                continue

            # 아직 쓰는 중인 파일이면 스킵
            if not stable_file(pcap, wait_sec=1.0):
                continue

            # pcap -> csv
            out_csv = csv_dir / (pcap.stem + ".csv")
            try:
                run_cicflowmeter(jar, pcap, out_csv)
            except Exception as e:
                print("[cicflowmeter fail]", pcap, e)
                seen.add(key)
                continue

            # csv 읽기 -> buffer에 누적
            try:
                rows = read_csv_rows(out_csv)
                buffer.extend(rows)
            except Exception as e:
                print("[csv read fail]", out_csv, e)

            seen.add(key)

            # batch 전송
            while len(buffer) >= args.batch_rows:
                chunk = buffer[:args.batch_rows]
                buffer = buffer[args.batch_rows:]
                try:
                    r = post_ingest(args.model_url, chunk)
                    print(f"[ingest] sent={len(chunk)} status={r.status_code} body={r.text[:200]}")
                except Exception as e:
                    print("[ingest fail]", e)
                    # 실패하면 chunk을 다시 앞에 붙여서 재시도(간단 복구)
                    buffer = chunk + buffer
                    time.sleep(2.0)
                    break

        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
