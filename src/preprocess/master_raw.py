import numpy as np
import pandas as pd
from pathlib import Path

# ============================================
# 0. 경로 설정
# ============================================

# 이 스크립트 파일 기준으로 프로젝트 루트를 잡는다.
ROOT_DIR = Path(__file__).resolve().parents[1]   # D:\NIDS

# 개별 GLF CSV들이 있는 폴더
INPUT_DIR = ROOT_DIR / "data" / "GeneratedLabelledFlows" / "TrafficLabelling"

# 마스터 CSV를 저장할 폴더/파일
OUTPUT_DIR = ROOT_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "master_raw.csv"

print("ROOT_DIR   :", ROOT_DIR)
print("INPUT_DIR  :", INPUT_DIR)
print("OUTPUT_PATH:", OUTPUT_PATH)


# ============================================
# 1. 폴더 안 CSV 파일 찾기
# ============================================

csv_files = sorted(INPUT_DIR.glob("*.csv"))
if not csv_files:
    raise RuntimeError(f"{INPUT_DIR} 안에서 .csv 파일을 찾지 못했음.")

print("발견된 CSV 파일:")
for f in csv_files:
    print("  -", f.name)


# ============================================
# 2. CSV 읽어서 하나의 DataFrame으로 concat
#    (컬럼명 공백 제거 + source_file 컬럼 추가)
# ============================================

dfs = []

for f in csv_files:
    print(f"\n[READ] {f}")
    df = pd.read_csv(
        f,
        encoding="latin1",   # CICIDS2017 GLF와 동일한 인코딩 (ISO-8859-1 계열)
        low_memory=False
    )

    # (1) 컬럼명 앞뒤 공백 제거
    #     예: ' Source IP' -> 'Source IP', ' Label' -> 'Label'
    df.columns = [c.strip() for c in df.columns]

    # (2) 원본 파일 이름 기록 (나중에 필요할 수 있어서)
    df["source_file"] = f.name

    dfs.append(df)

# 모든 요일/캠페인 CSV를 하나로 합치기
master = pd.concat(dfs, ignore_index=True)

print("\n=== concat 완료 ===")
print("총 행 수:", len(master))
print("총 컬럼 수:", len(master.columns))
print("컬럼 예시:", list(master.columns)[:15])


# ============================================
# 3. 규칙에 따라 컬럼 처리
#    - Flow ID 완전 제거
#    - Infinity 문자열 정리 (메모리 안전하게)
#    - 숫자/문자 컬럼 분리
# ============================================

# 3-1. Flow ID 완전 삭제
if "Flow ID" in master.columns:
    master = master.drop(columns=["Flow ID"])
    print("`Flow ID` 컬럼 삭제 완료")
else:
    print("경고: `Flow ID` 컬럼이 없어 삭제하지 못함")

# 3-2. Infinity / NaN 문자열 & 실제 inf 값 정리
#      → 메모리 절약을 위해 컬럼별로 처리

# (1) 우선 'object' 타입 컬럼에서 문자열 기반 Infinity/NaN 치환
obj_cols = master.select_dtypes(include="object").columns
print("\nobject 타입 컬럼 개수:", len(obj_cols))

for col in obj_cols:
    master[col] = master[col].replace(
        ["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan"],
        np.nan
    )

# (2) 숫자형 컬럼에서 실제 np.inf / -np.inf → np.nan
num_cols_all = master.select_dtypes(include=["number"]).columns
print("숫자 타입 컬럼(초기) 개수:", len(num_cols_all))

for col in num_cols_all:
    col_data = master[col].to_numpy()
    # isinf 체크 후 inf를 nan으로 바꿈
    mask = np.isinf(col_data)
    if mask.any():
        col_data[mask] = np.nan
        master[col] = col_data

# 3-3. 이후 단계에서 "숫자 스케일링/임베딩" 대상이 아닌 컬럼들
#      (IP / Port / Protocol / Timestamp / Label / source_file)
keep_as_str = [
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port",
    "Destination Port",
    "Protocol",
    "Label",
    "source_file",
]

# 실제로 존재하는 컬럼만 필터링
keep_as_str = [c for c in keep_as_str if c in master.columns]

# 나머지 = 숫자형으로 다뤄야 할 flow 통계 피처들
numeric_cols = [c for c in master.columns if c not in keep_as_str]

print("\n문자열(카테고리/식별자)로 유지할 컬럼들:")
print(keep_as_str)

print("\n숫자형으로 변환할 컬럼 개수:", len(numeric_cols))
print("예시:", numeric_cols[:15])

# 3-4. 숫자형 컬럼은 to_numeric + float32로 다운캐스팅 (메모리 절약)
for col in numeric_cols:
    master[col] = pd.to_numeric(master[col], errors="coerce").astype("float32")



# ============================================
# 4. 여기서는 "의도적으로" 하지 않는 것들
#    (이후 단계에서 반드시 처리해야 할 TODO)
# ============================================

# - NaN 채우기 (train 기준 median)
# - StandardScaler/RobustScaler fit & transform
# - Port/Protocol 카테고리 → index 부여 (임베딩용)
# - train/val/test split
# - dst IP 기준 윈도우링 (seq_len=128, stride=64, padding/drop)


# ============================================
# 5. 마스터 CSV 저장
# ============================================

master.to_csv(OUTPUT_PATH, index=False)
print(f"\n>>> 마스터 CSV 저장 완료: {OUTPUT_PATH}")
