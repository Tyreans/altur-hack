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


def decode_audio_b64(audio_b64: str) -> tuple[np.ndarray, int]:
    """
    Decodifica un audio (wav/flac/ogg) codificado en base64 a un
    arreglo mono float32 + su sample rate.

    Lanza ValueError si el payload no se puede decodificar, para que
    main.py lo convierta en un 400 limpio hacia el juez.
    """
    try:
        raw_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Base64 inválido: {e}")

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    except Exception as e:
        raise ValueError(f"No se pudo decodificar el audio: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # baja a mono si viene estéreo

    return audio, sample_rate