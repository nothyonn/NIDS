# src/sensor/pipeline.py
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def _norm_col(c: str) -> str:
    if c is None:
        return ""
    c = str(c).replace("\ufeff", "").strip()
    c = re.sub(r"\s+", " ", c)  # 연속 공백 정리
    return c


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _disambiguate_header_like_pandas(raw_header: List[str]) -> List[str]:
    """
    ⚠️ 중요:
    - DictReader는 중복 헤더를 dict 키로 만들 때 덮어써서 feature가 사라질 수 있음.
    - pandas는 중복 헤더를 .1, .2 ... 로 살려둠.
    => 센서도 pandas 방식으로 중복 헤더를 보존해서 학습 스키마와 동일하게 맞춤.
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in raw_header:
        c = _norm_col(c)
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out


def read_csv_rows(csv_path: Path) -> Tuple[List[Dict], List[str], str, List[str]]:
    """
    CICFlowMeter CSV를 row dict 리스트로 읽음(헤더 기반).
    returns: (rows, header_cols, header_hash, dup_bases)
    """
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        raw_header = reader.fieldnames or []
        header = _disambiguate_header_like_pandas(list(raw_header))

        dup_bases = sorted({c.rsplit(".", 1)[0] for c in header if re.search(r"\.\d+$", c)})

        # DictReader가 이 header를 사용하도록 강제
        reader.fieldnames = header

        for r in reader:
            rr = {(_norm_col(k) if k else k): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            rows.append(rr)

    header_hash = _sha256_text("\n".join(header))[:16]
    return rows, header, header_hash, dup_bases


def fetch_model_schema(model_url: str, timeout_sec: float = 3.0) -> Optional[Dict]:
    try:
        url = f"{model_url.rstrip('/')}/schema"
        r = requests.get(url, timeout=timeout_sec)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def schema_compare(header_cols: List[str], schema: Dict) -> Dict[str, object]:
    numeric_cols = schema.get("numeric_cols") or []
    norm_header = {_norm_col(c) for c in header_cols}
    norm_numeric = {_norm_col(c) for c in numeric_cols}

    inter = norm_header & norm_numeric
    missing = sorted(list(norm_numeric - norm_header))
    cov = (len(inter) / len(norm_numeric)) if norm_numeric else 0.0

    return {
        "coverage": float(cov),
        "missing_n": int(len(missing)),
        "missing_top": missing[:20],
        "schema_id": schema.get("schema_id") or (schema.get("config_sha256", "")[:12]),
    }


def post_ingest(
    model_url: str,
    flows: List[Dict],
    *,
    request_id: str,
    timeout_sec: int = 180,
) -> requests.Response:
    url = f"{model_url.rstrip('/')}/ingest"
    payload = {
        "flows": flows,
        "request_id": request_id,
        "drop_label": True,  # live flow엔 Label 없으니 드랍
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
    out_dir.mkdir(parents=True, exist_ok=True)
    work_in_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
        yield lst[i : i + n]


def cleanup_pcaps(
    *,
    target_dir: Path,
    retention_min: int,
    keep_last: int,
    log_fn,
    patterns: Tuple[str, ...] = ("*.pcap", "*.pcap.gz"),
) -> None:
    """
    target_dir에서 pcap을 정리:
    - keep_last > 0 이면 최신 N개는 무조건 보존
    - retention_min > 0 이면 mtime 기준으로 오래된 파일 삭제
    """
    if retention_min <= 0 and keep_last <= 0:
        return

    try:
        files: List[Path] = []
        for pat in patterns:
            files.extend(list(target_dir.glob(pat)))
        if not files:
            return

        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        keep_set = set()
        if keep_last > 0:
            for p in files[:keep_last]:
                keep_set.add(p.resolve())

        now = time.time()
        deleted = 0
        freed = 0

        for p in files:
            try:
                rp = p.resolve()
                if rp in keep_set:
                    continue
                st = p.stat()
                age_min = (now - st.st_mtime) / 60.0
                if retention_min > 0 and age_min <= float(retention_min):
                    continue
                size = st.st_size
                p.unlink(missing_ok=True)
                deleted += 1
                freed += size
            except FileNotFoundError:
                continue
            except Exception:
                continue

        if deleted:
            log_fn(
                f"cleanup_pcaps: dir={target_dir} deleted={deleted} freed={freed/1024/1024:.1f}MB "
                f"retention_min={retention_min} keep_last={keep_last}"
            )
    except Exception:
        return


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
    ap.add_argument("--schema-check", action="store_true", default=True, help="model /schema로 스키마 비교 로그(기본 ON)")
    ap.add_argument("--schema-timeout", type=float, default=3.0)

    # pipeline behavior
    ap.add_argument("--pcap-glob", default="*.pcap")
    ap.add_argument("--poll-sec", type=float, default=1.0)
    ap.add_argument("--stable-wait-sec", type=float, default=2.0)

    # micro-batch
    ap.add_argument("--batch-rows", type=int, default=256)
    ap.add_argument("--max-request-rows", type=int, default=2000)
    ap.add_argument("--flush-interval", type=float, default=2.0)
    ap.add_argument("--keep-csv", action="store_true")
    ap.add_argument("--keep-pcap", action="store_true")
    ap.add_argument("--once", action="store_true")

    # ✅ archive cleanup (pcap_done 폭주 방지)
    ap.add_argument("--archive-retention-min", type=int, default=5, help="pcap_done에서 이 분(min)보다 오래된 pcap 삭제(0이면 비활성)")
    ap.add_argument("--archive-keep-last", type=int, default=0, help="pcap_done 최신 N개는 보존(0이면 비활성)")
    ap.add_argument("--cleanup-interval-sec", type=float, default=30.0, help="cleanup 실행 주기(초)")

    # safety
    ap.add_argument("--seen-max", type=int, default=5000)
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

    # schema cache
    schema_cache: Optional[Dict] = None
    schema_cache_ts: float = 0.0
    schema_ttl = 60.0

    def get_schema_cached() -> Optional[Dict]:
        nonlocal schema_cache, schema_cache_ts
        if not args.schema_check:
            return None
        now = time.time()
        if schema_cache is not None and (now - schema_cache_ts) < schema_ttl:
            return schema_cache
        schema_cache = fetch_model_schema(args.model_url, timeout_sec=float(args.schema_timeout))
        schema_cache_ts = now
        return schema_cache

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
    log(f"schema_check={bool(args.schema_check)} schema_timeout={args.schema_timeout}s")
    log(f"cleanup: archive_retention_min={args.archive_retention_min} keep_last={args.archive_keep_last} interval={args.cleanup_interval_sec}s")

    next_cleanup_ts = 0.0

    while True:
        # ✅ 주기적으로 pcap_done 정리
        now0 = time.time()
        if now0 >= next_cleanup_ts:
            cleanup_pcaps(
                target_dir=archive_dir,
                retention_min=int(args.archive_retention_min),
                keep_last=int(args.archive_keep_last),
                log_fn=log,
            )
            next_cleanup_ts = now0 + float(args.cleanup_interval_sec)

        pcaps = sorted(pcap_dir.glob(args.pcap_glob))

        for pcap in pcaps:
            if pcap.name in seen_s:
                continue

            if not stable_file(pcap, wait_sec=float(args.stable_wait_sec)):
                continue

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

            schema = get_schema_cached()

            for csv_path in new_csvs:
                try:
                    rows, header_cols, header_hash, dup_bases = read_csv_rows(csv_path)

                    if schema:
                        cmp = schema_compare(header_cols, schema)
                        log(
                            f"schema_check: pcap={pcap.name} csv={csv_path.name} "
                            f"header_hash={header_hash} cov={cmp['coverage']:.3f} "
                            f"missing_n={cmp['missing_n']} schema_id={cmp['schema_id']} "
                            f"dup_bases={len(dup_bases)}"
                        )
                        if cmp["missing_n"]:
                            log(f"schema_missing_top: {cmp['missing_top']}")
                        for r in rows:
                            r["_debug_schema_cov"] = float(cmp["coverage"])
                            r["_debug_schema_missing_n"] = int(cmp["missing_n"])
                            r["_debug_schema_id"] = str(cmp["schema_id"])
                    else:
                        log(
                            f"schema_check: pcap={pcap.name} csv={csv_path.name} "
                            f"header_hash={header_hash} schema=NONE dup_bases={len(dup_bases)}"
                        )

                    now_ts = time.time()
                    for r in rows:
                        r["_debug_pcap"] = pcap.name
                        r["_debug_csv"] = csv_path.name
                        r["_debug_header_hash"] = header_hash
                        r["_debug_sensor_ts"] = now_ts

                    buffer.extend(rows)
                    log(f"csv read: {csv_path.name} rows={len(rows)} buffer={len(buffer)}")

                except Exception as e:
                    log(f"csv read FAIL: {csv_path.name} err={e}")

                if not args.keep_csv:
                    try:
                        csv_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            if not args.keep_pcap:
                try:
                    shutil.move(str(pcap), str(archive_dir / pcap.name))
                except Exception:
                    pass

            mark_seen(pcap.name)

        now = time.time()
        should_flush_by_time = (now - last_flush) >= float(args.flush_interval)
        should_flush_by_size = len(buffer) >= int(args.batch_rows)

        if buffer and (should_flush_by_size or should_flush_by_time):
            send_cap = int(args.max_request_rows)
            send_now = buffer[:send_cap]
            buffer = buffer[len(send_now) :]

            if should_flush_by_size and not should_flush_by_time:
                parts = list(chunked(send_now, int(args.batch_rows)))
            else:
                parts = [send_now]

            request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

            for part in parts:
                if not part:
                    continue
                for r in part:
                    r["_debug_request_id"] = request_id

                try:
                    r = post_ingest(args.model_url, part, request_id=request_id, timeout_sec=180)
                    log(f"ingest ok: req_id={request_id} sent={len(part)} status={r.status_code} body={r.text[:200]}")
                except Exception as e:
                    buffer = part + buffer
                    log(f"ingest FAIL: req_id={request_id} err={e} (requeued {len(part)})")
                    time.sleep(2.0)
                    break

            last_flush = now

        if args.once:
            if buffer:
                request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
                for r in buffer:
                    r["_debug_request_id"] = request_id
                try:
                    r = post_ingest(args.model_url, buffer, request_id=request_id, timeout_sec=180)
                    log(f"final flush ok: req_id={request_id} sent={len(buffer)} status={r.status_code} body={r.text[:200]}")
                except Exception as e:
                    log(f"final flush FAIL: req_id={request_id} err={e}")
            log("once mode done. exit.")
            return

        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    main()
