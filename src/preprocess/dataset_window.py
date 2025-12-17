import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
import math


def shannon_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class FlowWindowDataset(Dataset):
    """
    flow-level parquet → window-level dataset

    window-level features 추가됨:
        - unique_src_count
        - avg_flows_per_src
        - src_entropy
        - window_duration
        - flow_rate
    """

    def __init__(
        self,
        parquet_path: str,
        config_path: str,
        seq_len: int = 128,
        stride: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride

        # ----------------------------
        # Load config
        # ----------------------------
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.numeric_cols = cfg["numeric_cols"]               
        self.label_classes = cfg["label_classes"]
        self.num_classes = len(self.label_classes)

        self.cat_cols = ["sport_idx", "dport_idx", "proto_idx"]

        # ----------------------------
        # Load parquet
        # ----------------------------
        df = pd.read_parquet(parquet_path)

        # Timestamp 파싱
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"].astype(str).str.strip(),
            errors="coerce"
        )
        df["Timestamp"] = df["Timestamp"].fillna(pd.Timestamp("1970-01-01"))

        if "source_file" in df.columns:
            df = df.drop(columns=["source_file"])

        # ----------------------------
        # group by Destination IP
        # ----------------------------
        groups = {}
        for dst, g in df.groupby("Destination IP"):
            g = g.sort_values("Timestamp")
            groups[dst] = g.reset_index(drop=True)

        # ----------------------------
        # 윈도우 생성
        # ----------------------------
        self.windows = []
        for dst_ip, g in groups.items():
            self._extract_windows_from_group(g)

        print(f"[FlowWindowDataset] parquet={parquet_path}")
        print(f"[FlowWindowDataset] total windows = {len(self.windows)}")

    # ==========================================================
    # 윈도우 추출 + 윈도우-level feature 계산
    # ==========================================================
    def _extract_windows_from_group(self, g: pd.DataFrame):
        n = len(g)
        start = 0

        base_dim = len(self.numeric_cols)   # 77
        WINDOW_FEAT_DIM = 5                # 추가된 5개

        while start < n:
            end = start + self.seq_len
            win = g.iloc[start:end]
            real_len = len(win)

            if real_len == 0:
                break

            # ---------------- mask ----------------
            mask = np.zeros(self.seq_len, dtype=np.float32)
            mask[:real_len] = 1.0

            # ---------------- base numeric ----------------
            base_num = win[self.numeric_cols].to_numpy(dtype=np.float32)  # [real_len, 77]

            # ---------------- window-level feature 계산 ----------------
            total_flows = float(real_len)

            # 1) unique_src_count, src_counts
            if "Source IP" in win.columns:
                src_counts = win["Source IP"].value_counts()
                unique_src_count = float(len(src_counts))
            else:
                src_counts = np.array([], dtype=np.float32)
                unique_src_count = 1.0

            # 2) avg_flows_per_src
            if unique_src_count > 0:
                avg_flows_per_src = total_flows / unique_src_count
            else:
                avg_flows_per_src = 0.0

            # 3) src_entropy (Shannon)
            if unique_src_count > 0 and len(src_counts) > 0:
                src_ent_raw = shannon_entropy(src_counts.to_numpy(dtype=np.float32))
            else:
                src_ent_raw = 0.0

            # 4) window_duration (초)
            t_min = win["Timestamp"].min()
            t_max = win["Timestamp"].max()
            window_duration = float(max((t_max - t_min).total_seconds(), 1e-6))

            # 5) flow_rate
            flow_rate = total_flows / window_duration

            # ---------------- 간단 스케일링 ----------------
            # log1p로 크기 줄이고, 대충 0~1 근처 값이 되도록 상수 나눔

            # unique_src_count, avg_flows_per_src
            f1 = np.log1p(unique_src_count) / 5.0          # 대충 <= 1
            f2 = np.log1p(avg_flows_per_src) / 5.0

            # src_entropy 정규화: 0 ~ 1
            if unique_src_count > 1:
                max_ent = math.log2(unique_src_count)
                f3 = src_ent_raw / (max_ent + 1e-12)
                f3 = float(np.clip(f3, 0.0, 1.0))
            else:
                f3 = 0.0

            # window_duration, flow_rate도 log1p로 눌러서 0~1 근처
            f4 = np.log1p(window_duration) / 10.0          # 몇천 초 나와도 1 안 넘게
            f5 = np.log1p(flow_rate) / 5.0

            win_feat_vec = np.array([f1, f2, f3, f4, f5], dtype=np.float32)   # [5]

            # real_len timesteps에만 broadcast
            win_feat_real = np.repeat(win_feat_vec[None, :], real_len, axis=0)  # [real_len, 5]

            # base_num:[real_len,77], win_feat_real:[real_len,5] → [real_len,82]
            numeric_real = np.concatenate([base_num, win_feat_real], axis=1)

            # ---------------- padding (numeric) ----------------
            if real_len < self.seq_len:
                pad_len = self.seq_len - real_len
                pad_numeric = np.zeros(
                    (pad_len, base_dim + WINDOW_FEAT_DIM),
                    dtype=np.float32
                )
                num_arr = np.vstack([numeric_real, pad_numeric])  # [L, 82]
            else:
                num_arr = numeric_real  # [L, 82]

            # ---------------- categorical + padding ----------------
            if real_len < self.seq_len:
                pad_len = self.seq_len - real_len
                pad_cat = np.zeros((pad_len, len(self.cat_cols)), dtype=np.int64)
                cat_real = win[self.cat_cols].to_numpy(dtype=np.int64)
                cat_arr = np.vstack([cat_real, pad_cat])          # [L, 3]
            else:
                cat_arr = win[self.cat_cols].to_numpy(dtype=np.int64)

            # ---------------- multi-label target ----------------
            labels = win["Label"].unique().tolist()
            target = np.zeros(self.num_classes, dtype=np.float32)
            for lb in labels:
                if lb in self.label_classes:
                    idx = self.label_classes.index(lb)
                    target[idx] = 1.0

            self.windows.append(
                {
                    "numeric": num_arr,   # [seq_len, 77+5]
                    "cat": cat_arr,       # [seq_len, 3]
                    "mask": mask,         # [L]
                    "y": target,
                }
            )

            start += self.stride

    # ==========================================================
    # Dataset API
    # ==========================================================
    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        numeric = torch.tensor(item["numeric"], dtype=torch.float32)
        cat = torch.tensor(item["cat"], dtype=torch.int64)
        mask = torch.tensor(item["mask"], dtype=torch.float32)
        y = torch.tensor(item["y"], dtype=torch.float32)
        return numeric, cat, mask, y
