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
from concurrent.futures import ThreadPoolExecutor

from .acoustic_features import extract_acoustic_features
from .behavior_features import extract_behavior_features, get_vad_timestamps

_model_data = None

def load_model():
    global _model_data
    # Ruta absoluta robusta
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

    timings = {}
    
    # 1. Extraer VAD de Caller UNA vez
    start = time.perf_counter()
    caller_turns = get_vad_timestamps(caller_audio, sample_rate)
    agent_turns = get_vad_timestamps(agent_audio, sample_rate)
    
    # Recortar silencios para la fase acústica (Acelera Librosa enormemente)
    trimmed_caller = []
    for t in caller_turns:
        start_idx = int(t["start"] * sample_rate)
        end_idx = int(t["end"] * sample_rate)
        trimmed_caller.append(caller_audio[start_idx:end_idx])
    caller_audio_trimmed = np.concatenate(trimmed_caller) if trimmed_caller else caller_audio
    
    timings["VAD"] = time.perf_counter() - start

    # 2. Paralelizar extracción de features
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_acoustic = executor.submit(extract_acoustic_features, caller_audio_trimmed, sample_rate)
        f_behavior = executor.submit(extract_behavior_features, caller_audio, agent_audio, sample_rate, caller_turns, agent_turns)
        
        acoustic_feats = f_acoustic.result()
        behavior_feats = f_behavior.result()
        
    timings["Features"] = time.perf_counter() - start

    # 3. Imputación e Inferencia
    start = time.perf_counter()
    combined = {**acoustic_feats, **behavior_feats}
    df_infer = pd.DataFrame([combined])
    
    medians = _model_data.get("medians", {})
    feature_cols = _model_data["feature_cols"]
    
    # Aplicar lógica idéntica de NaN e imputación de train_classifier.py
    for col, val in medians.items():
        if f"{col}_was_missing" in feature_cols:
            df_infer[f"{col}_was_missing"] = 1.0 if pd.isna(df_infer.get(col, np.nan)[0]) else 0.0
        if col in df_infer.columns and pd.isna(df_infer[col][0]):
            df_infer[col] = val
            
    # Garantizar columnas y orden
    for col in feature_cols:
        if col not in df_infer:
            df_infer[col] = 0.0
            
    X_infer = df_infer[feature_cols]
    clf = _model_data["model"]
    confidence = float(clf.predict_proba(X_infer)[0, 1])
    timings["Infer"] = time.perf_counter() - start

    print(f"[evaluate_heavy] Latencia -> VAD: {timings['VAD']:.3f}s | Features: {timings['Features']:.3f}s | Infer: {timings['Infer']:.3f}s")
    return {"confidence_synthetic": confidence}