"""
Utilidad compartida: decodifica el audio base64 que manda el juez
a un arreglo numpy + sample rate, listo para pasarle a fast_checks
y heavy_model.

Nadie más debería necesitar tocar este archivo.
"""

import base64
import io

import numpy as np
import soundfile as sf


def decode_audio_b64(audio_b64: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Decodifica un WAV estéreo codificado en base64.

    Regresa (caller_audio, agent_audio, sample_rate):
        - caller_audio: canal 0, la voz a clasificar
        - agent_audio:  canal 1, el agente (contexto para comportamiento)
    """
    try:
        raw_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Base64 inválido: {e}")

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as e:
        raise ValueError(f"No se pudo decodificar el audio: {e}")

    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(
            f"Se esperaba WAV estéreo (canal 0 = caller, canal 1 = agente); "
            f"llegó shape={audio.shape}"
        )

    caller_audio = np.ascontiguousarray(audio[:, 0])
    agent_audio = np.ascontiguousarray(audio[:, 1])
    return caller_audio, agent_audio, sample_rate