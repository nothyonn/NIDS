# src/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from engine.service import FusionService
from engine.hec_client import SplunkHECConfig

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]  # NIDS 루트

# 너가 하드코딩 하든 env 쓰든 여기만 맞으면 됨
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
    drop_label: bool = True
    force_flush: bool = False
    min_flush_len: int = 16
    max_windows: int | None = None

@app.get("/health")
def health():
    return {"ok": True}

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
        splunk_cfg=splunk_cfg,
        drop_label=req.drop_label,
        force_flush=req.force_flush,
        min_flush_len=req.min_flush_len,
        max_windows=req.max_windows,
    )
