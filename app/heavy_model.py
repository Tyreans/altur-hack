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
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from .acoustic_features import extract_acoustic_features
from .behavior_features import extract_behavior_features, get_vad_timestamps

_model_data = None

def load_model():
    global _model_data
    model_path = Path(__file__).resolve().parent.parent / "model.joblib"
    
    if model_path.exists():
        _model_data = joblib.load(model_path)
        print(f"[heavy_model] Modelo RandomForest cargado desde {model_path}.")
    else:
        print(f"[heavy_model] WARNING: No se encontró {model_path}. Usando placeholder.")
        _model_data = "placeholder"

def evaluate_heavy(caller_audio: np.ndarray, agent_audio: np.ndarray, sample_rate: int) -> dict:
    if _model_data is None:
        raise RuntimeError("El modelo no se cargó. ¿Olvidaste llamar load_model()?")

    if _model_data == "placeholder":
        return {"confidence_synthetic": 0.5}

    t_start = time.perf_counter()
    
    # 1. Medir VAD
    caller_turns = get_vad_timestamps(caller_audio, sample_rate)
    agent_turns = get_vad_timestamps(agent_audio, sample_rate)
    t_vad = time.perf_counter()
    
    # 2. Medir Acústica (Aún sin recortar silencios, para ver el cuello de botella real)
    acoustic_feats = extract_acoustic_features(caller_audio, sample_rate)
    t_acoustic = time.perf_counter()
    
    # 3. Medir Conductual
    behavior_feats = extract_behavior_features(caller_audio, agent_audio, sample_rate, precomputed_caller_turns=caller_turns, precomputed_agent_turns=agent_turns)
    t_behavior = time.perf_counter()
    
    # 4. Imputación e Inferencia
    combined = {**acoustic_feats, **behavior_feats}
    df_infer = pd.DataFrame([combined])
    
    medians = _model_data.get("medians", {})
    feature_cols = _model_data["feature_cols"]
    
    for col, val in medians.items():
        if f"{col}_was_missing" in feature_cols:
            df_infer[f"{col}_was_missing"] = 1.0 if pd.isna(df_infer.get(col, np.nan)[0]) else 0.0
        if col in df_infer.columns and pd.isna(df_infer[col][0]):
            df_infer[col] = val
            
    for col in feature_cols:
        if col not in df_infer:
            df_infer[col] = 0.0
            
    X_infer = df_infer[feature_cols]
    clf = _model_data["model"]
    confidence = float(clf.predict_proba(X_infer)[0, 1])
    t_infer = time.perf_counter()

    print(f"[evaluate_heavy] Tiempos -> VAD: {t_vad-t_start:.3f}s | Acústica: {t_acoustic-t_vad:.3f}s | Conductual: {t_behavior-t_acoustic:.3f}s | Infer: {t_infer-t_behavior:.3f}s")
    
    return {"confidence_synthetic": confidence}