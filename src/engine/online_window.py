from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List


@dataclass
class WindowItem:
    dst_ip: str
    rows: List[Dict[str, Any]]
    real_len: int


class OnlineWindowBuffer:
    """
    dst_ip 기준 버퍼링 → seq_len/stride 조건 충족 시 window pop
    force_flush=True면 남은 것도 마지막 window로 내보냄(단, min_flush_len 이상일 때만)
    """

    def __init__(self, seq_len: int = 128, stride: int = 64, max_buffer: int = 5000):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.max_buffer = int(max_buffer)
        self.buf: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_buffer))

    def add_flows(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            dst = row.get("Destination IP") or row.get("dst_ip") or "unknown"
            self.buf[dst].append(row)

    def pop_windows(self, *, force_flush: bool = False, min_flush_len: int = 39) -> List[WindowItem]:
        out: List[WindowItem] = []

        for dst, dq in list(self.buf.items()):
            # normal pop
            while len(dq) >= self.seq_len:
                rows = [dq[i] for i in range(self.seq_len)]
                out.append(WindowItem(dst_ip=dst, rows=rows, real_len=self.seq_len))
                for _ in range(self.stride):
                    if dq:
                        dq.popleft()

            # flush
            if force_flush and len(dq) >= int(min_flush_len):
                rows = list(dq)
                out.append(WindowItem(dst_ip=dst, rows=rows, real_len=len(rows)))
                dq.clear()

            if not dq:
                self.buf.pop(dst, None)

        return out
