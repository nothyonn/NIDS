import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ========================================
# 0. 경로 설정
# ========================================
ROOT_DIR = Path(__file__).resolve().parents[2]  # D:\NIDS
INPUT_PATH = ROOT_DIR / "processed" / "master_raw.csv"

OUTPUT_DIR = ROOT_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_OUT = OUTPUT_DIR / "flows_train_scaled.parquet"
VAL_OUT   = OUTPUT_DIR / "flows_val_scaled.parquet"
TEST_OUT  = OUTPUT_DIR / "flows_test_scaled.parquet"
CONFIG_OUT = OUTPUT_DIR / "preprocess_config.json"

print("ROOT_DIR   :", ROOT_DIR)
print("INPUT_PATH :", INPUT_PATH)


# ========================================
# 1. master_raw 로드
# ========================================
print("\n[LOAD] master_raw.csv 로드 중...")
df = pd.read_csv(INPUT_PATH)
print(f"총 행 수: {len(df):,}")
print(f"컬럼 수: {df.shape[1]}")

if "Label" not in df.columns:
    raise RuntimeError("Label 컬럼이 없습니다. 전처리 중단.")


# ========================================
# 2. Port / Protocol 인덱스 생성
#    - 포트: 빈도 상위 N개 + 나머지 OTHER
#    - 프로토콜: 발견된 값 전부 인덱스 + 없으면 OTHER
# ========================================

# 숫자 변환 (에러는 NaN)
df["Source Port"] = pd.to_numeric(df["Source Port"], errors="coerce")
df["Destination Port"] = pd.to_numeric(df["Destination Port"], errors="coerce")
df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce")

# 2-1. Port 인덱스
print("\n[ENCODE] Port 인덱스 생성...")

all_ports = pd.concat([df["Source Port"], df["Destination Port"]], axis=0)
all_ports = all_ports.dropna().astype(int)

# 빈도 상위 N개 포트는 개별 카테고리, 나머지는 OTHER
TOP_N_PORTS = 15
top_ports = all_ports.value_counts().head(TOP_N_PORTS).index.tolist()
top_ports = sorted(top_ports)  # 정렬해서 인덱스 고정

port_idx_map = {}
current_idx = 0
for p in top_ports:
    port_idx_map[int(p)] = current_idx
    current_idx += 1

other_port_idx = current_idx  # 나머지 포트는 여기로

def map_port(p):
    try:
        p_int = int(p)
    except (TypeError, ValueError):
        return other_port_idx
    return port_idx_map.get(p_int, other_port_idx)

df["sport_idx"] = df["Source Port"].map(map_port)
df["dport_idx"] = df["Destination Port"].map(map_port)

print("Port 카테고리 개수 (including OTHER):", other_port_idx + 1)

# 2-2. Protocol 인덱스
print("\n[ENCODE] Protocol 인덱스 생성...")

protos = df["Protocol"].dropna().astype(int).value_counts().index.tolist()
protos = sorted(protos)

proto_idx_map = {}
for i, p in enumerate(protos):
    proto_idx_map[int(p)] = i

proto_other_idx = len(proto_idx_map)  # 혹시 없는 값 들어오면 OTHER

def map_proto(v):
    try:
        v_int = int(v)
    except (TypeError, ValueError):
        return proto_other_idx
    return proto_idx_map.get(v_int, proto_other_idx)

df["proto_idx"] = df["Protocol"].map(map_proto)

print("Protocol 카테고리 개수 (including OTHER):", proto_other_idx + 1)


# ========================================
# 3. Train / Val / Test Stratified Split
# ========================================
print("\n[SPLIT] Train 70% / Val 15% / Test 15% (Label 기준 Stratified)")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["Label"],
    random_state=42,
    shuffle=True,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["Label"],
    random_state=42,
    shuffle=True,
)

print(f"Train: {len(train_df):,}")
print(f"Val  : {len(val_df):,}")
print(f"Test : {len(test_df):,}")


# ========================================
# 4. Numeric feature 스케일링
#    - train 기준으로 median→NaN 채우기 → mean/std 계산
#    - Port/Protocol index, IP, Timestamp, Label 등은 스케일링 X
# ========================================

print("\n[SCALE] Numeric feature 스케일링 시작...")

# 스케일링에서 제외할 컬럼들
exclude_for_scaling = [
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "source_file",
    "Source Port",
    "Destination Port",
    "Protocol",
    "sport_idx",
    "dport_idx",
    "proto_idx",
]

# 숫자 타입 컬럼 중에서 제외대상 빼고 스케일링 대상 선정
num_candidates = train_df.select_dtypes(
    include=["int64", "float64", "float32", "Int64"]
).columns.tolist()

numeric_cols = [c for c in num_candidates if c not in exclude_for_scaling]

print("스케일링 대상 numeric 컬럼 수:", len(numeric_cols))
print("예시:", numeric_cols[:10])

# 4-A. NaN 채우기 (train median 기준)  🔥 추가된 부분
train_medians = train_df[numeric_cols].median()

train_df[numeric_cols] = train_df[numeric_cols].fillna(train_medians)
val_df[numeric_cols]   = val_df[numeric_cols].fillna(train_medians)
test_df[numeric_cols]  = test_df[numeric_cols].fillna(train_medians)

# train 기준 mean/std 계산
means = train_df[numeric_cols].mean()
stds = train_df[numeric_cols].std().replace(0, 1e-6)  # 0으로 나누기 방지

def apply_scaling(df_part: pd.DataFrame) -> pd.DataFrame:
    df_part = df_part.copy()
    df_part[numeric_cols] = (df_part[numeric_cols] - means) / stds
    return df_part

train_scaled = apply_scaling(train_df)
val_scaled = apply_scaling(val_df)
test_scaled = apply_scaling(test_df)


# ========================================
# 5. 결과 저장 (Parquet + 설정 JSON)
# ========================================

print("\n[SAVE] Parquet 저장 중...")
train_scaled.to_parquet(TRAIN_OUT, index=False)
val_scaled.to_parquet(VAL_OUT, index=False)
test_scaled.to_parquet(TEST_OUT, index=False)

print("저장 완료:")
print(" -", TRAIN_OUT)
print(" -", VAL_OUT)
print(" -", TEST_OUT)

print("\n[CONFIG] preprocess_config.json 저장 중...")

config = {
    "numeric_cols": numeric_cols,
    "port_idx_map": {str(k): int(v) for k, v in port_idx_map.items()},
    "other_port_idx": int(other_port_idx),
    "proto_idx_map": {str(k): int(v) for k, v in proto_idx_map.items()},
    "proto_other_idx": int(proto_other_idx),
    "scaler": {
        "mean": means.to_dict(),
        "std": stds.to_dict(),
    },
}

# 멀티라벨용 클래스 목록 (윈도우 타겟 만들 때 사용)
config["label_classes"] = sorted(df["Label"].unique().tolist())

with open(CONFIG_OUT, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(" -", CONFIG_OUT)
print("\n[DONE] 2차 전처리 완료.")
