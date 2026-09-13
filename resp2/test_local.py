"""
Prueba local para Miguel/Marco: corre fast_checks y heavy_model
directamente sobre un archivo de audio, SIN necesidad de levantar
FastAPI ni de mandar nada por red.

Uso:
    python test_local.py ruta/a/un_audio.wav
"""

import base64
import sys

from app import heavy_model
from app.audio_utils import decode_audio_b64
from app.fast_checks import evaluate_fast


def main(path: str):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    caller_audio, agent_audio, sr = decode_audio_b64(b64)
    print(f"Audio cargado: {len(caller_audio)} samples @ {sr}Hz")

    fast_result = evaluate_fast(caller_audio, sr)
    print("Fase 1 (rápida):", fast_result)

    heavy_model.load_model()
    heavy_result = heavy_model.evaluate_heavy(caller_audio, agent_audio, sr)
    print("Fase 2 (pesada):", heavy_result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python test_local.py <ruta_audio.wav>")
        sys.exit(1)
    main(sys.argv[1])