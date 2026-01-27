# src/engine/hec_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import os
import json
import math
import traceback

import requests


@dataclass
class SplunkHECConfig:
    url: str                 # e.g. "https://<SPLUNK>:8088/services/collector/event"
    token: str               # HEC token
    index: str = "main"
    sourcetype: str = "_json"
    host: str = "model-server"
    verify_tls: bool = False
    timeout_sec: int = 5

    # 추가: hec 전송 디버그 로그
    log_path: str = "/data/log/hec_client.log"
    log_body_limit: int = 600


def _now() -> float:
    return time.time()


def _log_line(path: str, obj: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _is_nonfinite(x: float) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def _sanitize_jsonable(v: Any) -> Any:
    """
    Splunk HEC에 넣기 전에 JSON 직렬화 / 비정상 수치(NaN/Inf) 방어.
    - NaN/Inf -> None
    - bytes -> utf-8 decode (실패 시 repr)
    - numpy scalar/array -> python scalar/list (numpy가 없어도 안전하게 처리)
    - pandas Timestamp/Datetime -> str
    """
    if v is None:
        return None

    # bool은 int보다 먼저
    if isinstance(v, bool):
        return v

    if isinstance(v, (int, str)):
        return v

    if isinstance(v, float):
        return None if _is_nonfinite(v) else v

    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return repr(v)

    # pandas.Timestamp 같은 것들: str로
    # (pandas import 없이도 안전하게)
    t = type(v).__name__
    if t in ("Timestamp", "datetime", "date"):
        try:
            return str(v)
        except Exception:
            return repr(v)

    # dict/list/tuple 재귀
    if isinstance(v, dict):
        out = {}
        for kk, vv in v.items():
            out[str(kk)] = _sanitize_jsonable(vv)
        return out

    if isinstance(v, (list, tuple)):
        return [_sanitize_jsonable(x) for x in v]

    # numpy 계열 (있으면 처리)
    try:
        import numpy as np  # type: ignore
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            fv = float(v)
            return None if _is_nonfinite(fv) else fv
        if isinstance(v, (np.ndarray,)):
            return [_sanitize_jsonable(x) for x in v.tolist()]
    except Exception:
        pass

    # 그 외: 문자열로 떨굼
    try:
        return str(v)
    except Exception:
        return repr(v)


class SplunkHECClient:
    def __init__(self, cfg: SplunkHECConfig):
        self.cfg = cfg

    def send_events(self, events: List[Dict[str, Any]]) -> int:
        """
        events: "event" 본문 dict 리스트
        return: 성공적으로 전송된 이벤트 수
        """
        headers_base = {
            "Authorization": f"Splunk {self.cfg.token}",
            "Content-Type": "application/json",
        }

        ok = 0

        for i, ev in enumerate(events):
            # request_id 추적(있으면 헤더에도 박아둠: splunk 내부로그에서 추적 쉬움)
            req_id = None
            try:
                req_id = ev.get("meta", {}).get("request_id")
            except Exception:
                req_id = None

            headers = dict(headers_base)
            if req_id:
                headers["X-Splunk-Request-Channel"] = str(req_id)

            safe_ev = _sanitize_jsonable(ev)

            payload = {
                "event": safe_ev,
                "host": self.cfg.host,
                "sourcetype": self.cfg.sourcetype,
                "index": self.cfg.index,
                "time": _now(),
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
                else:
                    _log_line(self.cfg.log_path, {
                        "ts": _now(),
                        "kind": "hec_http_error",
                        "i": i,
                        "status": r.status_code,
                        "url": self.cfg.url,
                        "req_id": req_id,
                        "resp": (r.text or "")[: self.cfg.log_body_limit],
                    })

            except Exception as e:
                _log_line(self.cfg.log_path, {
                    "ts": _now(),
                    "kind": "hec_exception",
                    "i": i,
                    "url": self.cfg.url,
                    "req_id": req_id,
                    "err": repr(e),
                    "trace": traceback.format_exc()[: 2000],
                })

        _log_line(self.cfg.log_path, {
            "ts": _now(),
            "kind": "hec_summary",
            "total": len(events),
            "ok": ok,
            "url": self.cfg.url,
            "index": self.cfg.index,
            "sourcetype": self.cfg.sourcetype,
        })

        return ok
