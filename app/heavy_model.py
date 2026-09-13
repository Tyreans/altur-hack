import os
import time
import joblib
import numpy as np
from pathlib import Path
import warnings

from .acoustic_features import extract_acoustic_features
from .behavior_features import extract_behavior_features, get_vad_timestamps

_model_data = None

def load_model() -> None:
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
        return {"confidence_synthetic": 0.5, "timings": {}}

    t0_total = time.perf_counter()
    timings = {}
    
    # 1. Medir VAD UNA SOLA VEZ para ambos canales
    t0_vad = time.perf_counter()
    caller_turns = get_vad_timestamps(caller_audio, sample_rate)
    timings["vad_ms"] = (time.perf_counter() - t0_vad) * 1000

    # 2. Medir Acústica pasándole los turnos precalculados del caller
    t0_acustica = time.perf_counter()
    acoustic_feats = extract_acoustic_features(caller_audio, sample_rate, caller_turns)
    
    # Limpieza de métricas internas del diccionario de características acústicas
    skipped_segs = acoustic_feats.pop("__skipped_short_segments", 0)
    timings["lfcc_ms"] = acoustic_feats.pop("__t_lfcc", 0.0)
    timings["parselmouth_ms"] = acoustic_feats.pop("__t_parselmouth", 0.0)
    timings["librosa_ms"] = acoustic_feats.pop("__t_librosa", 0.0)
    timings["acustica_ms"] = (time.perf_counter() - t0_acustica) * 1000
    timings["skipped_short_segments"] = skipped_segs
    
    # 3. Medir Conductual reciclando los turnos calculados en el paso 1
    t0_behav = time.perf_counter()
    behavior_feats = extract_behavior_features(
        caller_audio, 
        agent_audio, 
        sample_rate, 
        caller_turns=caller_turns,
    )
    timings["behavior_ms"] = (time.perf_counter() - t0_behav) * 1000
    
    # 4. Imputación e Inferencia SIN PANDAS (NumPy directo)
    t0_infer = time.perf_counter()
    combined = {**acoustic_feats, **behavior_feats}
    
    medians = _model_data.get("medians", {})
    feature_cols = _model_data["feature_cols"]
    
    infer_dict = {}
    for col, val_mediana in medians.items():
        base_val = combined.get(col, np.nan)
        
        # Bandera de valor faltante
        if f"{col}_was_missing" in feature_cols:
            infer_dict[f"{col}_was_missing"] = 1.0 if np.isnan(base_val) else 0.0
        
        # Imputación real
        infer_dict[col] = val_mediana if np.isnan(base_val) else base_val
            
    # Llenado de seguridad para columnas esperadas pero no generadas
    for col in feature_cols:
        if col not in infer_dict:
            infer_dict[col] = 0.0
            
   # Convertir directo a numpy array 2D
    X_infer = np.array([[infer_dict[col] for col in feature_cols]])
    
    clf = _model_data["model"]
    
    # Silenciamos la queja de Scikit-Learn sobre los feature names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        confidence = float(clf.predict_proba(X_infer)[0, 1])
        
    timings["inference_ms"] = (time.perf_counter() - t0_infer) * 1000
    timings["total_ms"] = (time.perf_counter() - t0_total) * 1000
    
    return {"confidence_synthetic": confidence, "timings": timings}