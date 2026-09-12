"""
Módulo de extracción de features conductuales.
Utiliza Silero VAD para detectar tiempos de habla en ambos canales
y computar métricas de interacción (latencia, interrupciones, backchanneling).
"""
import torch
import numpy as np
import warnings

# Suprimir warnings de PyTorch durante la carga de Silero
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Cargar modelo VAD globalmente para no recargarlo en cada llamada
    _vad_model, _utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False,
                                        trust_repo=True)
    get_speech_timestamps = _utils[0]


def get_vad_timestamps(audio: np.ndarray, sample_rate: int) -> list[dict]:
    """Retorna timestamps de habla [{start: secs, end: secs}, ...]"""
    # Silero VAD requiere float32 tensor
    tensor = torch.from_numpy(audio).float()
    
    # Silero VAD requiere al menos 512 muestras para 8kHz
    if len(tensor) < 512:
        return []
    
    # Parámetros por defecto para Silero. En 8kHz funciona bien.
    with torch.no_grad():
        timestamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=sample_rate)
    
    # Convertir a segundos
    return [{"start": ts["start"] / sample_rate, "end": ts["end"] / sample_rate} for ts in timestamps]


def extract_behavior_features(caller_audio: np.ndarray, agent_audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    """
    Extrae features conductuales cruzando los timestamps de ambos canales.
    """
    features = {
        "turn_gap_mean": 0.0,
        "turn_gap_var": 0.0,
        "cutoff_time_mean": 0.0,
        "cutoff_time_var": 0.0,
        "silence_reaction_latency_mean": 0.0,
        "backchannel_rate": 0.0
    }
    
    # Extraer turnos de cada canal
    caller_turns = get_vad_timestamps(caller_audio, sample_rate)
    agent_turns = get_vad_timestamps(agent_audio, sample_rate)
    
    # Calcular duración total del audio en segundos
    total_duration = len(caller_audio) / sample_rate
    if total_duration <= 0:
        return features
        
    # 1. Turn-taking gap (latencia): tiempo entre fin de turno de agente e inicio de turno de caller
    gaps = []
    for agent_turn in agent_turns:
        # Buscar el primer turno del caller que empiece DESPUÉS de que el agente terminó
        subsequent_caller_turns = [ct for ct in caller_turns if ct["start"] >= agent_turn["end"]]
        if subsequent_caller_turns:
            gap = subsequent_caller_turns[0]["start"] - agent_turn["end"]
            gaps.append(gap)
            
    if gaps:
        features["turn_gap_mean"] = float(np.mean(gaps))
        features["turn_gap_var"] = float(np.var(gaps))
        
    # 2. Cutoff time (tiempo de reacción a interrupción)
    # Cuando caller está hablando y agente empieza a hablar, ¿cuánto tarda caller en callar?
    cutoffs = []
    for ct in caller_turns:
        # Agente empieza a hablar durante el turno del caller
        interruptions = [at for at in agent_turns if ct["start"] < at["start"] < ct["end"]]
        for at in interruptions:
            cutoff = ct["end"] - at["start"]
            cutoffs.append(cutoff)
            
    if cutoffs:
        features["cutoff_time_mean"] = float(np.mean(cutoffs))
        features["cutoff_time_var"] = float(np.var(cutoffs))
        
    # 3. Silence reaction: reacción a silencios deliberados del agente (gaps largos)
    # Definamos gap largo como > 2 segundos entre turnos del agente
    long_silence_threshold = 2.0
    silence_reactions = []
    for i in range(len(agent_turns) - 1):
        silence_start = agent_turns[i]["end"]
        silence_end = agent_turns[i+1]["start"]
        if silence_end - silence_start > long_silence_threshold:
            # Caller habla en este silencio?
            caller_reactions = [ct for ct in caller_turns if silence_start < ct["start"] < silence_end]
            if caller_reactions:
                reaction_latency = caller_reactions[0]["start"] - silence_start
                silence_reactions.append(reaction_latency)
                
    if silence_reactions:
        features["silence_reaction_latency_mean"] = float(np.mean(silence_reactions))
        
    # 4. Backchanneling: turnos cortos del caller DURANTE turno del agente
    # Consideramos "corto" a turnos menores a 1.5 segundos
    backchannel_threshold = 1.5
    backchannels = 0
    for ct in caller_turns:
        duration = ct["end"] - ct["start"]
        if duration <= backchannel_threshold:
            # ¿Ocurre durante un turno del agente? (inicio de ct dentro de at)
            for at in agent_turns:
                if at["start"] <= ct["start"] <= at["end"]:
                    backchannels += 1
                    break # contar 1 vez
                    
    # Tasa por minuto para normalizar
    features["backchannel_rate"] = float(backchannels / (total_duration / 60))
    
    return features
