"""
Módulo de extracción de features acústicos.
Extrae LFCCs, Jitter, Shimmer y features espectrales de librosa
solo en los segmentos donde el caller está hablando (recorte por VAD).
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
from spafe.features.lfcc import lfcc
import librosa
import warnings
import time

def _get_default_features() -> dict:
    features = {}
    for i in range(13):
        features[f"lfcc_{i}_mean"] = 0.0
        features[f"lfcc_{i}_std"] = 0.0
        features[f"delta_lfcc_{i}_mean"] = 0.0
        features[f"delta_lfcc_{i}_std"] = 0.0
    features["jitter_local"] = 0.0
    features["shimmer_local"] = 0.0
    features["spectral_centroid_mean"] = 0.0
    features["spectral_centroid_std"] = 0.0
    features["spectral_flatness_mean"] = 0.0
    features["spectral_flatness_std"] = 0.0
    features["zcr_mean"] = 0.0
    features["zcr_std"] = 0.0
    return features

def _extract_segment_features(segment: np.ndarray, sample_rate: int) -> tuple[dict, dict]:
    """Extrae features y tiempos para un solo segmento de voz."""
    feats = _get_default_features()
    timings = {"t_lfcc": 0.0, "t_parselmouth": 0.0, "t_librosa": 0.0}
    
    t0 = time.perf_counter()
    # 1. LFCC con spafe
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lfccs = lfcc(segment, fs=sample_rate, num_ceps=13)
        delta_lfccs = librosa.feature.delta(lfccs.T).T
        for i in range(13):
            feats[f"lfcc_{i}_mean"] = float(np.mean(lfccs[:, i]))
            feats[f"lfcc_{i}_std"] = float(np.std(lfccs[:, i]))
            feats[f"delta_lfcc_{i}_mean"] = float(np.mean(delta_lfccs[:, i]))
            feats[f"delta_lfcc_{i}_std"] = float(np.std(delta_lfccs[:, i]))
    except Exception:
        pass
        
    t1 = time.perf_counter()
    timings["t_lfcc"] = (t1 - t0) * 1000

    # 2. Jitter y Shimmer usando parselmouth
    try:
        sound = parselmouth.Sound(segment.reshape(1, -1), sample_rate)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        local_jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        feats["jitter_local"] = local_jitter if not np.isnan(local_jitter) else 0.0
        
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        feats["shimmer_local"] = local_shimmer if not np.isnan(local_shimmer) else 0.0
    except Exception:
        pass
        
    t2 = time.perf_counter()
    timings["t_parselmouth"] = (t2 - t1) * 1000

    # 3. Spectral Features (librosa) - STFT compartido
    try:
        stft_matrix = np.abs(librosa.stft(segment))
        centroid = librosa.feature.spectral_centroid(S=stft_matrix, sr=sample_rate)[0]
        flatness = librosa.feature.spectral_flatness(S=stft_matrix)[0]
        zcr = librosa.feature.zero_crossing_rate(y=segment)[0]
        
        feats["spectral_centroid_mean"] = float(np.mean(centroid))
        feats["spectral_centroid_std"] = float(np.std(centroid))
        feats["spectral_flatness_mean"] = float(np.mean(flatness))
        feats["spectral_flatness_std"] = float(np.std(flatness))
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"] = float(np.std(zcr))
    except Exception:
        pass
        
    t3 = time.perf_counter()
    timings["t_librosa"] = (t3 - t2) * 1000

    return feats, timings

def extract_acoustic_features(caller_audio: np.ndarray, sample_rate: int, caller_turns: list[dict] | None = None) -> dict:
    features = _get_default_features()
    
    # Inicializar claves ocultas para el trackeo de tiempos
    features["__t_lfcc"] = 0.0
    features["__t_parselmouth"] = 0.0
    features["__t_librosa"] = 0.0
    features["__skipped_short_segments"] = 0
    
    if len(caller_audio) < sample_rate * 0.1 or np.all(caller_audio == 0):
        return features
        
    total_voice_duration = sum(t["end"] - t["start"] for t in caller_turns) if caller_turns else 0.0
    
    # Fallback: Si el audio original es muy corto o VAD detectó menos de 1s total,
    # procesamos el audio completo para evitar devolver puro silencio.
    if not caller_turns or total_voice_duration < 1.0:
        res_feats, res_timings = _extract_segment_features(caller_audio, sample_rate)
        res_feats["__t_lfcc"] = res_timings["t_lfcc"]
        res_feats["__t_parselmouth"] = res_timings["t_parselmouth"]
        res_feats["__t_librosa"] = res_timings["t_librosa"]
        res_feats["__skipped_short_segments"] = 0
        return res_feats

    # Extracción por segmentos 
    segment_features_list = []
    skipped_segments = 0
    sum_t_lfcc, sum_t_parselmouth, sum_t_librosa = 0.0, 0.0, 0.0
    
    for turn in caller_turns:
        duration = turn["end"] - turn["start"]
        # Evitamos pasar a Parselmouth segmentos minúsculos que romperán su rastreo de fase
        if duration < 0.1:
            skipped_segments += 1
            continue
            
        start_idx = int(turn["start"] * sample_rate)
        end_idx = int(turn["end"] * sample_rate)
        segment = caller_audio[start_idx:end_idx]
        
        seg_feats, seg_timings = _extract_segment_features(segment, sample_rate)
        segment_features_list.append((duration, seg_feats))
        
        sum_t_lfcc += seg_timings["t_lfcc"]
        sum_t_parselmouth += seg_timings["t_parselmouth"]
        sum_t_librosa += seg_timings["t_librosa"]
            
    if not segment_features_list:
        # Fallback extremo si TODOS los segmentos fueron ignorados por cortos
        res_feats, res_timings = _extract_segment_features(caller_audio, sample_rate)
        res_feats["__t_lfcc"] = res_timings["t_lfcc"]
        res_feats["__t_parselmouth"] = res_timings["t_parselmouth"]
        res_feats["__t_librosa"] = res_timings["t_librosa"]
        res_feats["__skipped_short_segments"] = skipped_segments
        return res_feats

    # Agregación ponderada (Segmentos de 8s pesan más que fragmentos de 0.2s)
    total_valid_duration = sum(w for w, _ in segment_features_list)
    keys = _get_default_features().keys()
    
    for k in keys:
        weighted_sum = sum(feats[k] * w for w, feats in segment_features_list)
        features[k] = weighted_sum / total_valid_duration

    # Inyectar métricas de depuración
    features["__t_lfcc"] = sum_t_lfcc
    features["__t_parselmouth"] = sum_t_parselmouth
    features["__t_librosa"] = sum_t_librosa
    features["__skipped_short_segments"] = skipped_segments
    
    return features