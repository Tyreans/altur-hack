from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    # AJUSTAR el nombre del campo en cuanto los organizadores confirmen
    # el formato exacto del payload. Esto es lo más importante que hay
    # que verificar hoy: si el campo se llama distinto, o si mandan
    # metadata adicional (sample_rate, formato, etc.), hay que reflejarlo
    # aquí de inmediato.
    audio_base64: str = Field(..., description="Audio codificado en base64 (wav/flac)")


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float