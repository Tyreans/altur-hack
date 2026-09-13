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
    
    if len(audio) == 0:
        return {"confidence_synthetic": 0.5, "signals": {"error": "audio vacio"}}

    # Heurística 1: Tasa de cruces por cero (ZCR) rápida (Proxy para textura metálica)
    # Las voces sintéticas a baja calidad suelen tener anomalías en los picos de ZCR.
    zcr = np.abs(np.diff(np.signbit(audio))).sum() / len(audio)
    signals["zcr_estimado"] = float(zcr)

    # Heurística 2: Dinámica de Energía RMS
    frame_len = max(int(sample_rate * 0.02), 1)
    frames = [audio[i:i + frame_len] for i in range(0, len(audio), frame_len)]
    energies = np.array([np.sqrt(np.mean(f ** 2)) for f in frames if len(f) > 0])
    
    energy_var = float(np.std(energies)) if len(energies) > 0 else 0.0
    signals["energy_variability"] = energy_var

    # Calibración rápida en base a ZCR y Energía
    # Voces muy monótonas (variabilidad baja) y ZCR excesivamente alto apuntan a vocoders IA.
    confidence = 0.5
    
    if energy_var < 0.005:
        confidence += 0.25  # Sospechoso: poco rango dinámico
    elif energy_var > 0.05:
        confidence -= 0.15  # Humano: rango dinámico rico
        
    if zcr > 0.15:
        confidence += 0.20  # Sospechoso: mucha fricción en frecuencias agudas
    
    # Limitar entre 0 y 1
    confidence = min(max(confidence, 0.0), 1.0)

    return {"confidence_synthetic": confidence, "signals": signals}