"""
Módulo de extracción de features acústicos.
Extrae LFCCs, Jitter, Shimmer y features espectrales de librosa.
"""
import numpy as np
import parselmouth
from parselmouth.praat import call
from spafe.features.lfcc import lfcc
import librosa
import warnings

def _get_default_features() -> dict[str, float]:
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

def extract_acoustic_features(caller_audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    features = _get_default_features()

    if len(caller_audio) < sample_rate * 0.1 or np.all(caller_audio == 0):
        return features

    # 1. LFCC con spafe
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lfccs = lfcc(caller_audio, fs=sample_rate, num_ceps=13)
        delta_lfccs = librosa.feature.delta(lfccs.T).T
        for i in range(lfccs.shape[1]):
            features[f"lfcc_{i}_mean"] = float(np.mean(lfccs[:, i]))
            features[f"lfcc_{i}_std"] = float(np.std(lfccs[:, i]))
            features[f"delta_lfcc_{i}_mean"] = float(np.mean(delta_lfccs[:, i]))
            features[f"delta_lfcc_{i}_std"] = float(np.std(delta_lfccs[:, i]))
    except Exception:
        pass # Usa defaults

    # 2. Jitter y Shimmer usando parselmouth
    try:
        sound = parselmouth.Sound(caller_audio.reshape(1, -1), sample_rate)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
        local_jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        features["jitter_local"] = local_jitter if not np.isnan(local_jitter) else np.nan
        
        local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features["shimmer_local"] = local_shimmer if not np.isnan(local_shimmer) else np.nan
    except Exception:
        features["jitter_local"] = np.nan
        features["shimmer_local"] = np.nan

    # 3. Spectral Features (librosa)
    try:
        centroid = librosa.feature.spectral_centroid(y=caller_audio, sr=sample_rate)[0]
        features["spectral_centroid_mean"] = float(np.mean(centroid))
        features["spectral_centroid_std"] = float(np.std(centroid))
        
        flatness = librosa.feature.spectral_flatness(y=caller_audio)[0]
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))
        
        zcr = librosa.feature.zero_crossing_rate(y=caller_audio)[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
    except Exception:
        pass

    return features