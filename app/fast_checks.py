"""
Fase 1: heurísticas rápidas de bajo costo.

Responsables: Miguel / Marco

CONTRATO (no rompan esta firma, main.py depende de ella):
    evaluate_fast(audio: np.ndarray, sample_rate: int) -> dict
        {
            "confidence_synthetic": float en [0, 1],
                # 1.0 = con toda seguridad es audio sintético/bot
                # 0.0 = con toda seguridad es un humano
            "signals": dict,  # valores intermedios libres, para debug/demo
        }

Pueden desarrollar y probar esta función SIN levantar FastAPI:
    python test_local.py ruta/a/un_audio.wav
"""

import numpy as np


def evaluate_fast(audio: np.ndarray, sample_rate: int) -> dict:
    signals = {}

    # --- Ejemplo 1: naturalidad del "piso de ruido" ---
    # El ruido de fondo real casi nunca es perfectamente plano; el audio
    # sintético a veces tiene silencios sospechosamente limpios.
    quiet_mask = np.abs(audio) < np.percentile(np.abs(audio), 10)
    noise_floor = np.abs(audio[quiet_mask])
    signals["noise_floor_std"] = float(np.std(noise_floor)) if len(noise_floor) else 0.0

    # --- Ejemplo 2: variabilidad de energía entre frames (proxy de
    # naturalidad de pausas / prosodia) ---
    frame_len = max(int(sample_rate * 0.02), 1)
    frames = [audio[i:i + frame_len] for i in range(0, len(audio), frame_len)]
    energies = np.array([np.sqrt(np.mean(f ** 2)) for f in frames if len(f) > 0])
    signals["energy_variability"] = float(np.std(energies)) if len(energies) else 0.0

    # TODO Miguel/Marco: esta combinación es un placeholder ingenuo, solo
    # para que el pipeline corra de punta a punta HOY. Reemplácenla por
    # algo calibrado contra el dataset real en cuanto puedan.
    score = 0.5
    if signals["noise_floor_std"] < 0.001:
        score += 0.3
    if signals["energy_variability"] < 0.01:
        score += 0.2
    score = min(max(score, 0.0), 1.0)

    return {"confidence_synthetic": score, "signals": signals}