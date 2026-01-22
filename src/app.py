# src/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from engine.service import FusionService
from engine.hec_client import SplunkHECConfig

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]  # NIDS 루트

SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "https://192.168.0.7:8088/services/collector")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "<토큰>")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "main")

svc = FusionService(
    root_dir=ROOT_DIR,
    config_rel="processed/preprocess_config.json",
    tcn_ckpt_rel="models/tcn_transformer_v2_best.pt",
    ae_ckpt_rel="models/ae_tcn_best.pt",
    seq_len=128,
    stride=64,
    ae_threshold=1.5624,
    ae_agg_mode="topk",
    ae_topk=3,
)

class IngestReq(BaseModel):
    flows: List[Dict[str, Any]]
    request_id: Optional[str] = None  # pipeline에서 주면 그대로, 없으면 model이 생성
    drop_label: bool = True
    force_flush: bool = False
    # 학습 규칙(패딩 70% 이상 drop) 맞춰 flush 최소 길이 39(=ceil(0.3*128)) 권장
    min_flush_len: int = 39
    max_windows: int | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/schema")
def schema():
    """센서가 실시간 CSV 헤더를 학습 스키마(numeric_cols)와 비교할 수 있도록 제공."""
    return svc.get_schema()

@app.post("/ingest")
def ingest(req: IngestReq):
    if not SPLUNK_HEC_TOKEN:
        return {"ok": False, "error": "SPLUNK_HEC_TOKEN is empty"}

    splunk_cfg = SplunkHECConfig(
        url=SPLUNK_HEC_URL,
        token=SPLUNK_HEC_TOKEN,
        index=SPLUNK_INDEX,
        sourcetype="_json",
        host="model-server",
        verify_tls=False,
        timeout_sec=5,
    )

    return svc.ingest_flows_and_send(
        flows=req.flows,
        request_id=req.request_id,
        splunk_cfg=splunk_cfg,
        drop_label=req.drop_label,
        force_flush=req.force_flush,
        min_flush_len=req.min_flush_len,
        max_windows=req.max_windows,
    )
