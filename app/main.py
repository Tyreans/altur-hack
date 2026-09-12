import asyncio

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from . import config, heavy_model
from .audio_utils import decode_audio_b64
from .fast_checks import evaluate_fast
from .schemas import DetectRequest, DetectResponse

app = FastAPI(title="Detector de Audio IA")


@app.on_event("startup")
async def startup_event():
    # Carga el modelo pesado UNA sola vez al arrancar, no por request.
    await run_in_threadpool(heavy_model.load_model)


@app.post("/detect", response_model=DetectResponse)
async def detect(payload: DetectRequest):
    try:
        audio, sample_rate = decode_audio_b64(payload.audio_base64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Fase 1 (rápida) — en threadpool para no bloquear el event loop
    fast_result = await run_in_threadpool(evaluate_fast, audio, sample_rate)
    conf = fast_result["confidence_synthetic"]

    if conf >= config.FAST_HIGH_THRESHOLD or conf <= config.FAST_LOW_THRESHOLD:
        return DetectResponse(
            is_synthetic=conf >= 0.5,
            confidence=conf,
            phase="fast",
            signals=fast_result["signals"],
        )

    # Fase 2 (pesada) — con timeout y fallback a la Fase 1 si falla
    try:
        heavy_result = await asyncio.wait_for(
            run_in_threadpool(heavy_model.evaluate_heavy, audio, sample_rate),
            timeout=config.HEAVY_MODEL_TIMEOUT,
        )
        return DetectResponse(
            is_synthetic=heavy_result["confidence_synthetic"] >= 0.5,
            confidence=heavy_result["confidence_synthetic"],
            phase="heavy",
            signals=fast_result["signals"],
        )
    except asyncio.TimeoutError:
        # No dejamos al juez sin respuesta: caemos de vuelta al veredicto
        # de la Fase 1 aunque fuera dudoso.
        return DetectResponse(
            is_synthetic=conf >= 0.5,
            confidence=conf,
            phase="fast_fallback",
            signals=fast_result["signals"],
        )


@app.get("/health")
async def health():
    return {"status": "ok"}