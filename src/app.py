# src/app.py
from __future__ import annotations

import os
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

from engine.service import FusionService, SplunkHECConfig

app = FastAPI()

ROOT_DIR = Path(__file__).resolve().parents[1]  # 프로젝트 루트(NIDS) 기준으로 맞춰짐

# ---- Splunk HEC (환경변수로 받는 걸 권장) ----
SPLUNK_HEC_URL = os.getenv("SPLUNK_HEC_URL", "http://192.168.8.129:8088/services/collector")
SPLUNK_HEC_TOKEN = os.getenv("SPLUNK_HEC_TOKEN", "")
SPLUNK_INDEX = os.getenv("SPLUNK_INDEX", "main")

# ---- 서비스 초기화 (서버 시작 시 1번 로드) ----
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

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/replay")
def replay(req: ReplayReq):
    if not SPLUNK_HEC_TOKEN:
        return {
            "ok": False,
            "error": "SPLUNK_HEC_TOKEN env var is empty. Set it first.",
            "hint": "export SPLUNK_HEC_TOKEN='xxxxx' (and optionally SPLUNK_HEC_URL)",
        }

    splunk = SplunkHECConfig(
        url=SPLUNK_HEC_URL,
        token=SPLUNK_HEC_TOKEN,
        index=SPLUNK_INDEX,
        sourcetype="_json",
        host="model-server",
        verify_tls=False,
        timeout_sec=3,
    )

    summary = svc.infer_parquet_and_send(
        parquet_rel=req.parquet_rel,
        splunk=splunk,
        max_batches=req.max_batches,
        sleep_sec=req.sleep_sec,
    )
    return {"ok": True, "summary": summary}
