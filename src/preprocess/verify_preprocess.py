import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 경로 설정
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]  # D:\NIDS
PROCESSED_DIR = ROOT_DIR / "processed"

MASTER_PATH = PROCESSED_DIR / "master_raw.csv"
TRAIN_PATH  = PROCESSED_DIR / "flows_train_scaled.parquet"
VAL_PATH    = PROCESSED_DIR / "flows_val_scaled.parquet"
TEST_PATH   = PROCESSED_DIR / "flows_test_scaled.parquet"
CONFIG_PATH = PROCESSED_DIR / "preprocess_config.json"

print("ROOT_DIR     :", ROOT_DIR)
print("PROCESSED_DIR:", PROCESSED_DIR)
print("MASTER_PATH  :", MASTER_PATH)
print("TRAIN_PATH   :", TRAIN_PATH)
print("VAL_PATH     :", VAL_PATH)
print("TEST_PATH    :", TEST_PATH)
print("CONFIG_PATH  :", CONFIG_PATH)


# ============================================================
# 1. master_raw.csv 검증
# ============================================================
print("\n========== [1] master_raw.csv 검증 ==========")

master = pd.read_csv(MASTER_PATH, low_memory=False)
print(f"- master_raw 행/열: {master.shape[0]:,} rows, {master.shape[1]} cols")

if "Label" not in master.columns:
    raise RuntimeError("master_raw.csv에 Label 컬럼이 없습니다.")

label_nan_count = master["Label"].isna().sum()
print(f"- Label NaN 개수: {label_nan_count}")

print("- Label 값 예시:", master["Label"].dropna().unique()[:10])

# 전체 NaN 컬럼 상위
nan_per_col_master = master.isna().sum().sort_values(ascending=False).head(10)
print("\n- NaN 상위 10개 컬럼 (master):")
print(nan_per_col_master)


# ============================================================
# 2. parquet(train/val/test) 로드 및 기본 정보
# ============================================================
print("\n========== [2] parquet(train/val/test) 기본 정보 ==========")

train = pd.read_parquet(TRAIN_PATH)
val   = pd.read_parquet(VAL_PATH)
test  = pd.read_parquet(TEST_PATH)

print(f"- train shape: {train.shape[0]:,} rows, {train.shape[1]} cols")
print(f"- val   shape: {val.shape[0]:,} rows, {val.shape[1]} cols")
print(f"- test  shape: {test.shape[0]:,} rows, {test.shape[1]} cols")

print("\n- 컬럼 목록:")
print(train.columns.tolist())


# ============================================================
# 3. dtype / NaN / Label 분포 체크
# ============================================================
print("\n========== [3] dtype / NaN / Label 분포 체크 ==========")

print("\n[train] dtypes:")
print(train.dtypes.sort_index())

print("\n[train] NaN 상위 10개 컬럼:")
print(train.isna().sum().sort_values(ascending=False).head(10))

print("\n[val] NaN 상위 10개 컬럼:")
print(val.isna().sum().sort_values(ascending=False).head(10))

print("\n[test] NaN 상위 10개 컬럼:")
print(test.isna().sum().sort_values(ascending=False).head(10))

# Label 분포
if "Label" not in train.columns:
    raise RuntimeError("train 데이터에 Label 컬럼이 없습니다.")

print("\n[train] Label 분포 (비율):")
print(train["Label"].value_counts(normalize=True))

print("\n[val] Label 분포 (비율):")
print(val["Label"].value_counts(normalize=True))

print("\n[test] Label 분포 (비율):")
print(test["Label"].value_counts(normalize=True))


# ============================================================
# 4. Port / Protocol 인덱스 분포 확인
#    (second_preprocess.py에서 이름이 'sport_idx', 'dport_idx', 'proto_idx'임)
# ============================================================
print("\n========== [4] Port / Protocol 인덱스 분포 ==========")

for col in ["sport_idx", "dport_idx", "proto_idx"]:
    if col not in train.columns:
        print(f"- 경고: {col} 컬럼이 train에 없습니다.")
    else:
        uniq = np.sort(train[col].unique())
        print(f"- {col} unique 값 개수: {len(uniq)}")
        print(f"  예시: {uniq[:20]}")


# ============================================================
# 5. preprocess_config.json 검증
# ============================================================
print("\n========== [5] preprocess_config.json 검증 ==========")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

print("- config 키 목록:", list(cfg.keys()))

numeric_cols      = cfg.get("numeric_cols", [])
port_idx_map      = cfg.get("port_idx_map", {})
other_port_idx    = cfg.get("other_port_idx", None)
proto_idx_map     = cfg.get("proto_idx_map", {})
proto_other_idx   = cfg.get("proto_other_idx", None)
scaler_cfg        = cfg.get("scaler", {})
scaler_mean       = scaler_cfg.get("mean", {})
scaler_std        = scaler_cfg.get("std", {})

print(f"- numeric_cols 개수: {len(numeric_cols)}")
print(f"- port_idx_map 크기: {len(port_idx_map)} (OTHER idx = {other_port_idx})")
print(f"- proto_idx_map 크기: {len(proto_idx_map)} (OTHER idx = {proto_other_idx})")

print("label_classes: ", cfg["label_classes"])

missing_numeric_in_train = [c for c in numeric_cols if c not in train.columns]
if missing_numeric_in_train:
    print("\n❌ train에 없는 numeric_cols:")
    print(missing_numeric_in_train)
else:
    print("\n✅ numeric_cols는 모두 train에 존재")


# ============================================================
# 6. 스케일링 검증: train의 numeric_cols mean/std 확인
# ============================================================
print("\n========== [6] 스케일링 검증 (train 기준) ==========")

if numeric_cols:
    # train에서 실제 mean/std 재계산
    train_num = train[numeric_cols]

    actual_mean = train_num.mean()
    actual_std  = train_num.std()

    # config에 저장된 mean/std와 차이 비교
    cfg_mean = pd.Series(scaler_mean)
    cfg_std  = pd.Series(scaler_std)

    # config에 없는 컬럼은 제외
    common_cols = [c for c in numeric_cols if (c in cfg_mean.index and c in cfg_std.index)]
    diff_mean = (actual_mean[common_cols] - cfg_mean[common_cols]).abs()
    diff_std  = (actual_std[common_cols] - cfg_std[common_cols]).abs()

    print("- mean 차이 (상위 10개):")
    print(diff_mean.sort_values(ascending=False).head(10))

    print("\n- std 차이 (상위 10개):")
    print(diff_std.sort_values(ascending=False).head(10))

    print("\n- mean 차이 최대값:", float(diff_mean.max()))
    print("- std  차이 최대값:", float(diff_std.max()))
else:
    print("- numeric_cols 정보가 config에 없습니다. (스케일링 검증 불가)")


# ============================================================
# 7. numeric feature 분포 sanity check
# ============================================================
print("\n========== [7] numeric feature 분포 (sanity check) ==========")

if numeric_cols:
    desc = train[numeric_cols].describe().T
    print(desc[["mean", "std", "min", "max"]].head(15))
else:
    print("- numeric_cols 없음, 분포 출력 생략")


print("\n✅ verify_preprocess.py: 검증 완료")
