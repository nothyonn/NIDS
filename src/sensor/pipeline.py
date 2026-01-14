# src/sensor/pipeline.py
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

import requests


def stable_file(path: Path, wait_sec: float = 2.0) -> bool:
    """파일 크기가 wait_sec 동안 변하지 않으면 닫힌 것으로 간주."""
    try:
        s1 = path.stat().st_size
        time.sleep(wait_sec)
        s2 = path.stat().st_size
        return s1 == s2 and s1 > 0
    except FileNotFoundError:
        return False


def read_csv_rows(csv_path: Path) -> List[Dict]:
    """CICFlowMeter CSV를 row dict 리스트로 읽음(헤더 기반)."""
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rr = {(k.strip() if k else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append(rr)
    return rows


def post_ingest(model_url: str, flows: List[Dict], timeout_sec: int = 180) -> requests.Response:
    url = f"{model_url.rstrip('/')}/ingest"
    payload = {
        "flows": flows,
        "run_max_batches": 999999,  # 모델이 알아서 윈도잉/추론
        "drop_label": True,         # live flow엔 Label 없으니 드랍
    }
    return requests.post(url, json=payload, timeout=timeout_sec)


def run_cicflowmeter_v3_for_one_pcap(
    *,
    cic_jar: Path,
    jnet_dir: Path,
    pcap_file: Path,
    out_dir: Path,
    work_in_dir: Path,
    log_path: Path,
) -> List[Path]:
    """
    CICFlowMeter V3는 "pcap 디렉터리"를 입력으로 받음.
    파일 1개만 처리하려면 work_in_dir에 그 파일만 넣고 돌리는 게 가장 안전.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_in_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # work dir 정리 -> 파일 1개만 투입
    for x in work_in_dir.glob("*.pcap"):
        x.unlink(missing_ok=True)

    tmp_pcap = work_in_dir / pcap_file.name
    shutil.copy2(pcap_file, tmp_pcap)

    before = set(out_dir.glob("*.csv"))

    cmd = [
        "java",
        f"-Djava.library.path={str(jnet_dir)}",
        "-cp",
        str(cic_jar),
        "cic.cs.unb.ca.ifm.CICFlowMeter",
        str(work_in_dir) + "/",  # trailing slash 중요
        str(out_dir) + "/",      # trailing slash 중요
    ]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{str(jnet_dir)}:{env.get('LD_LIBRARY_PATH','')}"

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"\n=== CICFlowMeter run: {pcap_file} @ {time.strftime('%F %T')} ===\n")
        lf.flush()
        subprocess.run(cmd, check=True, env=env, stdout=lf, stderr=lf)

    after = set(out_dir.glob("*.csv"))
    new_csvs = sorted(after - before)
    return new_csvs


def chunked(lst: List[Dict], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    ap = argparse.ArgumentParser()

    # dirs
    ap.add_argument("--pcap-dir", default="/data/pcap")
    ap.add_argument("--flows-dir", default="/data/flows")
    ap.add_argument("--pcap-archive-dir", default="/data/pcap_done")
    ap.add_argument("--work-in-dir", default="/data/_cic_in")
    ap.add_argument("--log", default="/data/log/pipeline.log")
    ap.add_argument("--cic-log", default="/data/log/cicflowmeter.log")

    # CICFlowMeter V3
    ap.add_argument("--cic-jar", default="/opt/CICFlowMeter/target/CICFlowMeterV3-0.0.4-SNAPSHOT.jar")
    ap.add_argument("--jnet-dir", default="/opt/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425")

    # model
    ap.add_argument("--model-url", required=True)

    # pipeline behavior
    ap.add_argument("--pcap-glob", default="*.pcap")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--stable-wait-sec", type=float, default=2.0, help="pcap 파일 닫힘 판단 대기(권장 2.0)")

    # micro-batch
    ap.add_argument("--batch-rows", type=int, default=256, help="size flush 기준 (권장 256)")
    ap.add_argument("--max-request-rows", type=int, default=2000, help="1회 요청 body 최대 flow 수")
    ap.add_argument("--flush-interval", type=float, default=2.0, help="time flush 기준 (권장 2.0)")
    ap.add_argument("--keep-csv", action="store_true", help="csv 파일을 /data/flows에 남김(디버깅용)")
    ap.add_argument("--keep-pcap", action="store_true", help="pcap 파일을 archive로 옮기지 않고 남김(비권장)")
    ap.add_argument("--once", action="store_true", help="현재 pcap들 처리하고 종료")

    # safety
    ap.add_argument("--seen-max", type=int, default=5000, help="seen 목록 최대 크기(무한 증가 방지)")
    args = ap.parse_args()

    pcap_dir = Path(args.pcap_dir)
    flows_dir = Path(args.flows_dir)
    archive_dir = Path(args.pcap_archive_dir)
    work_in_dir = Path(args.work_in_dir)
    log_path = Path(args.log)
    cic_log_path = Path(args.cic_log)

    cic_jar = Path(args.cic_jar)
    jnet_dir = Path(args.jnet_dir)

    pcap_dir.mkdir(parents=True, exist_ok=True)
    flows_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    work_in_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not cic_jar.exists():
        raise FileNotFoundError(f"CICFlowMeter jar not found: {cic_jar}")
    if not jnet_dir.exists():
        raise FileNotFoundError(f"jnetpcap dir not found: {jnet_dir}")

    # seen을 무한히 키우지 않기 위해 deque+set 조합
    seen_q = deque(maxlen=int(args.seen_max))
    seen_s: Set[str] = set()

    def mark_seen(name: str):
        if name in seen_s:
            return
        if len(seen_q) == seen_q.maxlen:
            old = seen_q.popleft()
            seen_s.discard(old)
        seen_q.append(name)
        seen_s.add(name)

    buffer: List[Dict] = []
    last_flush = time.time()

    def log(msg: str) -> None:
        ts = time.strftime("%F %T")
        line = f"[{ts}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("sensor pipeline start")
    log(f"pcap_dir={pcap_dir} flows_dir={flows_dir} archive_dir={archive_dir}")
    log(f"cic_jar={cic_jar}")
    log(f"model_url={args.model_url}")
    log(f"batch_rows={args.batch_rows} flush_interval={args.flush_interval}s max_request_rows={args.max_request_rows}")
    log(f"poll_sec={args.poll_sec} stable_wait_sec={args.stable_wait_sec}")

    while True:
        pcaps = sorted(pcap_dir.glob(args.pcap_glob))

        for pcap in pcaps:
            if pcap.name in seen_s:
                continue

            # 아직 쓰는 중이면 skip
            if not stable_file(pcap, wait_sec=float(args.stable_wait_sec)):
                continue

            # pcap -> csv
            try:
                new_csvs = run_cicflowmeter_v3_for_one_pcap(
                    cic_jar=cic_jar,
                    jnet_dir=jnet_dir,
                    pcap_file=pcap,
                    out_dir=flows_dir,
                    work_in_dir=work_in_dir,
                    log_path=cic_log_path,
                )
                log(f"cicflowmeter ok: {pcap.name} -> new_csvs={len(new_csvs)}")
            except Exception as e:
                log(f"cicflowmeter FAIL: {pcap.name} err={e}")
                mark_seen(pcap.name)
                continue

            # csv 읽어서 buffer 누적
            for csv_path in new_csvs:
                try:
                    rows = read_csv_rows(csv_path)
                    buffer.extend(rows)
                    log(f"csv read: {csv_path.name} rows={len(rows)} buffer={len(buffer)}")
                except Exception as e:
                    log(f"csv read FAIL: {csv_path.name} err={e}")

                # csv 유지/삭제
                if not args.keep_csv:
                    try:
                        csv_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            # pcap 이동/유지
            if not args.keep_pcap:
                try:
                    shutil.move(str(pcap), str(archive_dir / pcap.name))
                except Exception:
                    pass

            mark_seen(pcap.name)

        # ---- flush logic 개선 ----
        now = time.time()
        should_flush_by_time = (now - last_flush) >= float(args.flush_interval)
        should_flush_by_size = len(buffer) >= int(args.batch_rows)

        if buffer and (should_flush_by_size or should_flush_by_time):
            # 이번 flush에서 보낼 최대량 (너무 커지면 지연/413 위험)
            send_cap = int(args.max_request_rows)
            send_now = buffer[:send_cap]
            buffer = buffer[len(send_now):]

            # 1) size flush: 안정적으로 batch_rows로 쪼개서 여러 번 전송
            # 2) time flush: 실시간성을 위해 "남은 것까지" 전송 (batch_rows에 안 맞아도 보냄)
            if should_flush_by_size and not should_flush_by_time:
                parts = list(chunked(send_now, int(args.batch_rows)))
            else:
                # time flush면 통째로 보내되, send_cap은 이미 적용됨
                parts = [send_now]

            for part in parts:
                if not part:
                    continue
                try:
                    r = post_ingest(args.model_url, part, timeout_sec=180)
                    log(f"ingest ok: sent={len(part)} status={r.status_code} body={r.text[:200]}")
                except Exception as e:
                    # 실패 시 복구: part를 앞에 다시 붙임
                    buffer = part + buffer
                    log(f"ingest FAIL: err={e} (requeued {len(part)})")
                    time.sleep(2.0)
                    break

            last_flush = now

        if args.once:
            # 남은 buffer도 한번 더 flush 시도하고 종료
            if buffer:
                try:
                    r = post_ingest(args.model_url, buffer, timeout_sec=180)
                    log(f"final flush ok: sent={len(buffer)} status={r.status_code} body={r.text[:200]}")
                except Exception as e:
                    log(f"final flush FAIL: err={e}")
            log("once mode done. exit.")
            return

        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    main()
