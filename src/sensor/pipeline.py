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
    중요:
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
    - 여기서 절대 값 전처리(스케일링/NaN/Infinity 처리) 하지 않음.
    returns: (rows, header_cols, header_hash, dup_bases)
    """
    rows: List[Dict] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        raw_header = reader.fieldnames or []
        header = _disambiguate_header_like_pandas(list(raw_header))

        dup_bases = sorted({c.rsplit(".", 1)[0] for c in header if re.search(r"\.\d+$", c)})

        # DictReader가 이 header를 사용하도록 강제 (중복키 덮어쓰기 방지)
        reader.fieldnames = header

        for r in reader:
            rr: Dict = {}
            for k, v in r.items():
                kk = _norm_col(k) if k else k
                if isinstance(v, str):
                    rr[kk] = v.strip()
                else:
                    rr[kk] = v
            # 빈 행 제거(가끔 CICFlowMeter가 마지막에 빈 줄 뱉음)
            if any(v not in (None, "", " ") for v in rr.values()):
                rows.append(rr)

    header_hash = _sha256_text("\n".join(header))[:16]
    return rows, header, header_hash, dup_bases


def fetch_model_schema(model_url: str, timeout_sec: float = 3.0, *, verify_ssl: bool = False) -> Optional[Dict]:
    try:
        url = f"{model_url.rstrip('/')}/schema"
        r = requests.get(url, timeout=timeout_sec, verify=verify_ssl)
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
    verify_ssl: bool = False,
) -> requests.Response:
    url = f"{model_url.rstrip('/')}/ingest"
    payload = {
        "flows": flows,
        "request_id": request_id,
        "drop_label": True,  # live flow엔 Label 없으니 드랍
    }
    return requests.post(url, json=payload, timeout=timeout_sec, verify=verify_ssl)


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
    너의 CICFlowMeterV3 실행 방식 유지.
    - work_in_dir 에 pcap을 복사하고
    - CICFlowMeter가 work_in_dir 전체를 처리해서 out_dir에 csv 생성
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work_in_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 이전 workdir pcap 제거 (혼합 방지)
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
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.flush()
        subprocess.run(cmd, check=True, env=env, stdout=lf, stderr=lf)

    after = set(out_dir.glob("*.csv"))
    new_csvs = sorted(after - before)

    # 만약 파일명이 overwrite되어 set diff가 0이 되는 경우 대비(최후방어):
    if not new_csvs:
        # 최근 수정된 csv 몇 개를 주워온다(매우 보수적으로 3개)
        candidates = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        new_csvs = candidates[:3]

    return new_csvs


def chunked(lst: List[Dict], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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

    # schema check: 기본 ON, 필요하면 --no-schema-check로 끔
    ap.add_argument("--no-schema-check", action="store_true", help="model /schema 비교 로그 끄기")
    ap.add_argument("--schema-timeout", type=float, default=3.0)

    # SSL verify (기본 False: 내부망 self-signed 환경에서 경고/실패 방지)
    ap.add_argument("--verify-ssl", action="store_true", help="requests SSL verify 켜기(기본 OFF)")

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

    # safety
    ap.add_argument("--seen-max", type=int, default=5000)

    args = ap.parse_args()

    verify_ssl = bool(args.verify_ssl)
    schema_check = not bool(args.no_schema_check)

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
        if not schema_check:
            return None
        now = time.time()
        if schema_cache is not None and (now - schema_cache_ts) < schema_ttl:
            return schema_cache
        schema_cache = fetch_model_schema(args.model_url, timeout_sec=float(args.schema_timeout), verify_ssl=verify_ssl)
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
    log(f"schema_check={bool(schema_check)} schema_timeout={args.schema_timeout}s verify_ssl={verify_ssl}")

    while True:
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

            base_request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"

            for part_idx, part in enumerate(parts):
                if not part:
                    continue

                # part별 request_id (base 유지 + suffix)
                request_id = f"{base_request_id}-p{part_idx}"

                for r in part:
                    r["_debug_request_id"] = request_id

                try:
                    resp = post_ingest(args.model_url, part, request_id=request_id, timeout_sec=180, verify_ssl=verify_ssl)
                    log(f"ingest ok: req_id={request_id} sent={len(part)} status={resp.status_code} body={resp.text[:200]}")
                except Exception as e:
                    buffer = part + buffer
                    log(f"ingest FAIL: req_id={request_id} err={e} (requeued {len(part)})")
                    time.sleep(2.0)
                    break

            last_flush = now

        if args.once:
            if buffer:
                request_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}-final"
                for r in buffer:
                    r["_debug_request_id"] = request_id
                try:
                    resp = post_ingest(args.model_url, buffer, request_id=request_id, timeout_sec=180, verify_ssl=verify_ssl)
                    log(f"final flush ok: req_id={request_id} sent={len(buffer)} status={resp.status_code} body={resp.text[:200]}")
                except Exception as e:
                    log(f"final flush FAIL: req_id={request_id} err={e}")
            log("once mode done. exit.")
            return

        time.sleep(float(args.poll_sec))


if __name__ == "__main__":
    main()
