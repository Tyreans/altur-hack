"""
Fase 2: modelo pesado (HuggingFace).

Responsables: Miguel / Marco

CONTRATO (no rompan estas firmas, main.py depende de ellas):
    load_model() -> None
        Se llama UNA sola vez, cuando arranca el servidor (main.py lo
        invoca en el evento de startup). Aquí se carga el modelo en
        memoria/GPU para que cada request NO tenga que recargarlo.

    evaluate_heavy(audio: np.ndarray, sample_rate: int) -> dict
        {
            "confidence_synthetic": float en [0, 1],
        }

Pueden desarrollar y probar esto SIN levantar FastAPI:
    python test_local.py ruta/a/un_audio.wav

Corran esto en la laptop con GPU (Marco/Miguel) mientras desarrollan;
al final basta con que este archivo funcione, se integra solo.
"""

import numpy as np

_model = None


def load_model():
    global _model
    # TODO: reemplazar por el modelo real, por ejemplo:
    #
    #   from transformers import pipeline
    #   _model = pipeline("audio-classification", model="<repo/modelo>")
    #
    _model = "placeholder-model-loaded"
    print("[heavy_model] Modelo cargado (placeholder).")


def evaluate_heavy(audio: np.ndarray, sample_rate: int) -> dict:
    if _model is None:
        raise RuntimeError("El modelo no se cargó. ¿Olvidaste llamar load_model()?")

    # TODO: inferencia real, por ejemplo:
    #
    #   result = _model({"array": audio, "sampling_rate": sample_rate})
    #   confidence = mapear_salida_a_0_1(result)
    #
    confidence = 0.5  # placeholder
    return {"confidence_synthetic": confidence}