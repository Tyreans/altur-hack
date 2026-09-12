"""
Fase 2: modelo pesado (ahora usa el Random Forest entrenado).

Responsables: Miguel / Marco

CONTRATO (no rompan estas firmas, main.py depende de ellas):
    load_model() -> None
        Se llama UNA sola vez, cuando arranca el servidor (main.py lo
        invoca en el evento de startup). Aquí se carga el modelo en
        memoria/GPU para que cada request NO tenga que recargarlo.

    evaluate_heavy(caller_audio: np.ndarray, agent_audio: np.ndarray, sample_rate: int) -> dict
        {
            "confidence_synthetic": float en [0, 1],
        }

Pueden desarrollar y probar esto SIN levantar FastAPI:
    python test_local.py ruta/a/un_audio.wav
"""

import os
import joblib
import pandas as pd
import numpy as np

from .acoustic_features import extract_acoustic_features
from .behavior_features import extract_behavior_features

_model_data = None


def load_model():
    global _model_data
    model_path = "model.joblib"
    if os.path.exists(model_path):
        _model_data = joblib.load(model_path)
        print(f"[heavy_model] Modelo RandomForest cargado desde {model_path}.")
    else:
        print(f"[heavy_model] WARNING: No se encontró {model_path}. Usando placeholder.")
        _model_data = "placeholder"


def evaluate_heavy(caller_audio: np.ndarray, agent_audio: np.ndarray, sample_rate: int) -> dict:
    if _model_data is None:
        raise RuntimeError("El modelo no se cargó. ¿Olvidaste llamar load_model()?")

    if _model_data == "placeholder":
        # Fallback si no han entrenado el modelo todavía
        return {"confidence_synthetic": 0.5}

    # 1. Extraer features de ambos módulos
    acoustic_feats = extract_acoustic_features(caller_audio, sample_rate)
    behavior_feats = extract_behavior_features(caller_audio, agent_audio, sample_rate)
    
    combined = {**acoustic_feats, **behavior_feats}

    # 2. Convertir a DataFrame y asegurar el orden exacto de las columnas de entrenamiento
    feature_cols = _model_data["feature_cols"]
    X_infer = pd.DataFrame([combined])[feature_cols].fillna(0.0)

    # 3. Inferencia
    clf = _model_data["model"]
    confidence = float(clf.predict_proba(X_infer)[0, 1])

    return {"confidence_synthetic": confidence}