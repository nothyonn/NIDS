# src/app.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from engine.service import FusionService, SplunkHECConfig

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]  # NIDS 루트

SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "http://192.168.8.129:8088/services/collector")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "main")

svc = FusionService(
    root_dir=ROOT_DIR,
    config_rel="processed/preprocess_config.json",
    tcn_ckpt_rel="models/tcn_transformer_v2_best.pt",
    ae_ckpt_rel="models/ae_tcn_best.pt",
    seq_len=128,
    stride=64,
    batch_size=64,
    ae_threshold=1.5624,
    ae_agg_mode="topk",
    ae_topk=3,
)

class ReplayReq(BaseModel):
    parquet_rel: str = "processed/flows_test_scaled.parquet"
    max_batches: int = 10
    sleep_sec: float = 0.0

class IngestReq(BaseModel):
    flows: List[Dict[str, Any]]
    run_max_batches: int = 10
    drop_label: bool = False

@app.get("/health")
def health():
    return {"ok": True}

def _make_splunk():
    return SplunkHECConfig(
        url=SPLUNK_HEC_URL,
        token=SPLUNK_HEC_TOKEN,
        index=SPLUNK_INDEX,
        sourcetype="_json",
        host="model-server",
        verify_tls=False,
        timeout_sec=3,
    )

@app.post("/replay")
def replay(req: ReplayReq):
    if not SPLUNK_HEC_TOKEN:
        return {"ok": False, "error": "SPLUNK_HEC_TOKEN env var is empty."}

    summary = svc.infer_parquet_and_send(
        parquet_rel=req.parquet_rel,
        splunk=_make_splunk(),
        max_batches=req.max_batches,
        sleep_sec=req.sleep_sec,
    )
    return {"ok": True, "summary": summary}

@app.post("/ingest")
def ingest(req: IngestReq):
    if not SPLUNK_HEC_TOKEN:
        return {"ok": False, "error": "SPLUNK_HEC_TOKEN env var is empty."}

    if not req.flows:
        return {"ok": False, "msg": "empty flows"}

    df = pd.DataFrame(req.flows)
    if req.drop_label and "Label" in df.columns:
        df = df.drop(columns=["Label"])

    tmp_path = Path("/tmp/ingest_live.parquet")
    df.to_parquet(tmp_path, index=False)

    summary = svc.infer_parquet_and_send(
        parquet_path=str(tmp_path),     # service.py에서 지원해야 함
        splunk=_make_splunk(),
        max_batches=req.run_max_batches,
        sleep_sec=0.0,
    )
    return {"ok": True, "summary": summary, "tmp_parquet": str(tmp_path)}
