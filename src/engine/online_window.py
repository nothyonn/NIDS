# src/engine/online_window.py
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class WindowItem:
    dst_ip: str
    rows: List[Dict[str, Any]]   # real rows only (len == real_len)
    real_len: int


class OnlineWindowBuffer:
    """
    dst_ip 기준 버퍼링 → seq_len/stride 조건 충족 시 window pop

    - add_flows(): raw/preprocessed row(dict)들을 넣는다
    - pop_windows(): seq_len 충족된 것 pop
    - force_flush=True: seq_len 미만이라도 min_flush_len 이상이면 1개 window로 pop(실시간/종료용)
    """

    def __init__(self, seq_len: int = 128, stride: int = 64, max_buffer: int = 5000):
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.max_buffer = int(max_buffer)
        self.buf = defaultdict(lambda: deque(maxlen=self.max_buffer))  # dst_ip -> deque(rows)

    def add_flows(self, flows: List[Dict[str, Any]]) -> None:
        for row in flows:
            dst = row.get("Destination IP") or row.get("dst_ip") or "unknown"
            self.buf[dst].append(row)

    def pop_windows(
        self,
        *,
        force_flush: bool = False,
        min_flush_len: int = 16,
    ) -> List[WindowItem]:
        out: List[WindowItem] = []

        for dst, dq in list(self.buf.items()):
            n = len(dq)
            if n <= 0:
                continue

            # 1) 정상 pop (seq_len 충족)
            while len(dq) >= self.seq_len:
                rows = [dq[i] for i in range(self.seq_len)]
                out.append(WindowItem(dst_ip=dst, rows=rows, real_len=self.seq_len))

                # stride만큼만 앞으로 당김
                for _ in range(self.stride):
                    if dq:
                        dq.popleft()

            # 2) flush pop (seq_len 미만 잔여)
            if force_flush and len(dq) >= int(min_flush_len):
                rows = list(dq)
                out.append(WindowItem(dst_ip=dst, rows=rows, real_len=len(rows)))

                # overlap 유지할지/완전 비울지 선택
                # 실시간에서는 overlap 유지가 좋지만, flush는 보통 "마무리"라 비우는게 편함
                dq.clear()

            # 버퍼가 비면 dict도 정리
            if len(dq) == 0:
                self.buf.pop(dst, None)

        return out
