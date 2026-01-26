# src/sensor/schema_diff.py
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


def norm_col(s: str) -> str:
    """헤더/컬럼명 정규화: BOM 제거, strip, 다중공백 축약, 제어문자 제거"""
    if s is None:
        return ""
    s = str(s)

    # BOM 제거
    s = s.replace("\ufeff", "")

    # 제어문자 제거
    s = "".join(ch for ch in s if ch.isprintable())

    # strip + 공백 축약
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def try_read_header(csv_path: Path, encodings: List[str]) -> Tuple[List[str], str]:
    last_err = None
    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, errors="strict", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
            return header, enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV header read failed for {csv_path} encodings={encodings} last_err={last_err}")


def load_config(cfg_path: Path) -> Dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="예: src/sensor/schema_test.pcap_ISCX.csv")
    ap.add_argument("--cfg", required=True, help="예: processed/preprocess_config.json")
    ap.add_argument("--outdir", required=True, help="예: src/sensor/schema_diff_out")
    ap.add_argument("--topk", type=int, default=8, help="각 missing에 대해 후보 몇 개 보여줄지")
    ap.add_argument("--cutoff", type=float, default=0.70, help="후보로 인정할 유사도 컷")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    cfg_path = Path(args.cfg).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    cfg = load_config(cfg_path)

    # config에서 기대 컬럼(여기선 numeric_cols 중심)
    expected = list(cfg.get("numeric_cols", []))
    # 모델쪽에서 raw로도 필요할 수 있는 컬럼들(참고용)
    expected_extra = [
        "Timestamp", "Source IP", "Destination IP",
        "Source Port", "Destination Port", "Protocol",
        "sport_idx", "dport_idx", "proto_idx", "Label", "source_file",
    ]

    expected_all = expected + expected_extra
    expected_all_norm = [norm_col(x) for x in expected_all]

    # CSV 헤더 읽기: utf-8-sig 먼저, 안 되면 utf-8, cp949 순으로
    header_raw, used_enc = try_read_header(csv_path, ["utf-8-sig", "utf-8", "cp949"])
    header_norm = [norm_col(x) for x in header_raw]

    # 빠른 lookup
    have_set = set(header_norm)

    # 1) exact match / missing
    missing = []
    exact_match = []
    for want_raw, want in zip(expected_all, expected_all_norm):
        if want in have_set:
            exact_match.append({"expected": want_raw, "matched": want, "match_type": "EXACT"})
        else:
            missing.append(want_raw)

    # 2) missing에 대해 후보 제안
    candidates = []
    for m in missing:
        m_norm = norm_col(m)
        scored = []
        for h_raw, h_norm in zip(header_raw, header_norm):
            sc = similarity(m_norm, h_norm)
            if sc >= float(args.cutoff):
                scored.append((sc, h_raw, h_norm))
        scored.sort(reverse=True, key=lambda x: x[0])

        candidates.append({
            "missing_col": m,
            "match_type": "UNMATCHED" if not scored else "CANDIDATES",
            "candidates": [
                {"score": round(sc, 4), "csv_col_raw": hr, "csv_col_norm": hn}
                for sc, hr, hn in scored[: int(args.topk)]
            ],
        })

    # 3) CSV에만 있는 extra
    expected_norm_set = set(expected_all_norm)
    extra = []
    for h_raw, h_norm in zip(header_raw, header_norm):
        if h_norm and h_norm not in expected_norm_set:
            extra.append(h_raw)

    # 저장(전부 UTF-8)
    (outdir / "meta.txt").write_text(
        f"csv={csv_path}\ncfg={cfg_path}\nused_encoding={used_enc}\n"
        f"csv_cols={len(header_raw)} expected_cols={len(expected_all)}\n"
        f"missing={len(missing)} extra={len(extra)} exact={len(exact_match)}\n",
        encoding="utf-8",
    )
    (outdir / "csv_header_raw.txt").write_text("\n".join(header_raw), encoding="utf-8")
    (outdir / "csv_header_norm.txt").write_text("\n".join(header_norm), encoding="utf-8")
    (outdir / "missing_all.txt").write_text("\n".join(missing), encoding="utf-8")
    (outdir / "extra_all.txt").write_text("\n".join(extra), encoding="utf-8")

    with (outdir / "match_results.json").open("w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    # CSV summary
    with (outdir / "match_results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["missing_col", "match_type", "score", "candidate_raw", "candidate_norm"])
        for row in candidates:
            m = row["missing_col"]
            mt = row["match_type"]
            if not row["candidates"]:
                w.writerow([m, mt, "", "", ""])
            else:
                for c in row["candidates"]:
                    w.writerow([m, mt, c["score"], c["csv_col_raw"], c["csv_col_norm"]])

    print("[OK] wrote to:", outdir)
    print((outdir / "meta.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
