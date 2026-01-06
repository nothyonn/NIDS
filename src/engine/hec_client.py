# src/engine/hec_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time

import requests


@dataclass
class SplunkHECConfig:
    url: str                 # e.g. "https://192.168.8.129:8088/services/collector"
    token: str               # HEC token
    index: str = "main"
    sourcetype: str = "_json"
    host: str = "model-server"
    verify_tls: bool = False
    timeout_sec: int = 5


class SplunkHECClient:
    def __init__(self, cfg: SplunkHECConfig):
        self.cfg = cfg

    def send_events(self, events: List[Dict[str, Any]]) -> int:
        """
        events: "event" 본문 dict 리스트
        return: 성공적으로 전송된 이벤트 수
        """
        headers = {
            "Authorization": f"Splunk {self.cfg.token}",
            "Content-Type": "application/json",
        }

        ok = 0
        for ev in events:
            payload = {
                "event": ev,
                "host": self.cfg.host,
                "sourcetype": self.cfg.sourcetype,
                "index": self.cfg.index,
                "time": time.time(),
            }
            try:
                r = requests.post(
                    self.cfg.url,
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.timeout_sec,
                    verify=self.cfg.verify_tls,
                )
                if 200 <= r.status_code < 300:
                    ok += 1
            except Exception:
                pass
        return ok
