# src/engine/online_window.py
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple

import pandas as pd


class OnlineWindowBuffer:
    """
    센서에서 들어온 flow row(dict)를 dst_ip 기준으로 버퍼링하고,
    seq_len/stride 조건이 되면 window(DataFrame)를 꺼낸다.

    NOTE: 지금은 parquet 리플레이 테스트에 안 쓰고,
          다음 단계(센서 JSON ingest)에서 사용.
    """

    def __init__(self, seq_len: int = 128, stride: int = 64, max_buffer: int = 5000):
        self.seq_len = seq_len
        self.stride = stride
        self.max_buffer = max_buffer
        self.buf = defaultdict(lambda: deque(maxlen=max_buffer))  # dst_ip -> deque(rows)

    def add_flows(self, flows: List[Dict[str, Any]]) -> None:
        for row in flows:
            dst = row.get("Destination IP") or row.get("dst_ip")
            if not dst:
                continue
            self.buf[dst].append(row)

    def pop_windows(self) -> List[Tuple[str, pd.DataFrame]]:
        out: List[Tuple[str, pd.DataFrame]] = []
        for dst, dq in list(self.buf.items()):
            if len(dq) < self.seq_len:
                continue

            df = pd.DataFrame(list(dq))
            n = len(df)

            idx = 0
            while idx + self.seq_len <= n:
                w = df.iloc[idx: idx + self.seq_len].copy()
                out.append((dst, w))
                idx += self.stride

            # overlap 유지: 마지막 (seq_len - stride)만 남김
            keep = max(0, self.seq_len - self.stride)
            remain = df.iloc[-keep:].to_dict("records") if keep else []
            self.buf[dst] = deque(remain, maxlen=self.max_buffer)

        return out
