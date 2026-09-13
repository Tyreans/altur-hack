import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. ENFORCING SINGLE-THREADED LINEAR ALGEBRA
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import asyncio
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
import numpy as np

from . import config, heavy_model
from .audio_utils import decode_audio_b64
from .fast_checks import evaluate_fast
from .schemas import DetectRequest, DetectResponse

torch.set_num_threads(1)

app = FastAPI(title="Detector de Audio IA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite cualquier origen (ideal para el prototipo/hackathon)
    allow_credentials=True,
    allow_methods=["*"], # Permite OPTIONS, POST, GET, etc.
    allow_headers=["*"], # Permite cualquier header (incluyendo application/json)
)
heavy_semaphore = None

@app.on_event("startup")
async def startup_event():
    global heavy_semaphore
    heavy_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_HEAVY)
    
    await run_in_threadpool(heavy_model.load_model)
    
    print("Calentando motores del pipeline acústico...")
    dummy_audio = np.zeros(8000, dtype="float32")
    try:
        await run_in_threadpool(heavy_model.evaluate_heavy, dummy_audio, dummy_audio, 8000)
        print("Pipeline listo y caliente. Ya puedes lanzar peticiones.")
    except Exception as e:
        print(f"Nota de warmup: {e}")

@app.post("/detect", response_model=DetectResponse)
async def detect_audio(payload: DetectRequest):
    try:
        caller_audio, agent_audio, sample_rate = decode_audio_b64(payload.audio_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    fast_result = await run_in_threadpool(evaluate_fast, caller_audio, sample_rate)
    conf = fast_result["confidence_synthetic"]

    # Salida por fast_checks
    if conf >= config.FAST_HIGH_THRESHOLD or conf <= config.FAST_LOW_THRESHOLD:
        return DetectResponse(
            is_synthetic=conf >= 0.5,
            confidence=conf
        )
    
    try:
        async def process_heavy_queued():
            async with heavy_semaphore:
                return await run_in_threadpool(
                    heavy_model.evaluate_heavy, caller_audio, agent_audio, sample_rate
                )
        
        heavy_result = await asyncio.wait_for(
            process_heavy_queued(), 
            timeout=config.HEAVY_MODEL_TIMEOUT
        )
        
        # Salida exitosa de modelo pesado
        return DetectResponse(
            is_synthetic=heavy_result["confidence_synthetic"] >= 0.5,
            confidence=heavy_result["confidence_synthetic"]
        )
        
    except asyncio.TimeoutError:
        # Salida por fallback de tiempo
        return DetectResponse(
            is_synthetic=conf >= 0.5,
            confidence=conf
        )
    except Exception as e:
        print(f"[main.py] Fallo en evaluate_heavy (Fallback a Fase 1): {e}")
        # Salida por error en modelo pesado
        return DetectResponse(
            is_synthetic=conf >= 0.5,
            confidence=conf
        )

@app.get("/health")
async def health():
    return {"status": "ok"}