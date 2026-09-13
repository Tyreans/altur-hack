"""
Módulo de extracción de features conductuales.
Utiliza Silero VAD para detectar tiempos de habla en ambos canales
y computar métricas de interacción (latencia, interrupciones, backchanneling).
"""
import torch
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # ONNX=True reduce drásticamente la latencia de inferencia en CPU
    _vad_model, _utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        onnx=True,
                                        force_reload=False,
                                        trust_repo=True)
    get_speech_timestamps = _utils[0]

def get_vad_timestamps(audio: np.ndarray, sample_rate: int) -> list[dict]:
    tensor = torch.from_numpy(audio).float()
    if len(tensor) < 512:
        return []
    with torch.no_grad():
        timestamps = get_speech_timestamps(tensor, _vad_model, sampling_rate=sample_rate)
    return [{"start": ts["start"] / sample_rate, "end": ts["end"] / sample_rate} for ts in timestamps]

def extract_behavior_features(caller_audio: np.ndarray, agent_audio: np.ndarray, sample_rate: int, 
                              precomputed_caller_turns=None, precomputed_agent_turns=None) -> dict[str, float]:
    features = {
        "turn_gap_mean": 0.0,
        "turn_gap_var": 0.0,
        "cutoff_time_mean": 0.0,
        "cutoff_time_var": 0.0,
        "silence_reaction_latency_mean": 0.0,
        "backchannel_rate": 0.0
    }
    
    caller_turns = precomputed_caller_turns if precomputed_caller_turns is not None else get_vad_timestamps(caller_audio, sample_rate)
    agent_turns = precomputed_agent_turns if precomputed_agent_turns is not None else get_vad_timestamps(agent_audio, sample_rate)
    
    total_duration = len(caller_audio) / sample_rate
    if total_duration <= 0:
        return features
        
    # 1. Turn-taking gap corregido
    gaps = []
    for agent_turn in agent_turns:
        subsequent_caller_turns = [ct for ct in caller_turns if ct["start"] >= agent_turn["end"]]
        if subsequent_caller_turns:
            next_ct = subsequent_caller_turns[0]
            # Verificar interrupciones intermedias del agente
            intermediate_agent = [at for at in agent_turns if agent_turn["end"] < at["start"] < next_ct["start"]]
            if not intermediate_agent:
                gaps.append(next_ct["start"] - agent_turn["end"])
                
    if gaps:
        features["turn_gap_mean"] = float(np.mean(gaps))
        features["turn_gap_var"] = float(np.var(gaps))
        
    # 2. Cutoff time
    cutoffs = []
    for ct in caller_turns:
        interruptions = [at for at in agent_turns if ct["start"] < at["start"] < ct["end"]]
        for at in interruptions:
            cutoffs.append(ct["end"] - at["start"])
            
    if cutoffs:
        features["cutoff_time_mean"] = float(np.mean(cutoffs))
        features["cutoff_time_var"] = float(np.var(cutoffs))
        
    # 3. Silence reaction (calibrado a 2.5s)
    long_silence_threshold = 2.5
    silence_reactions = []
    for i in range(len(agent_turns) - 1):
        silence_start = agent_turns[i]["end"]
        silence_end = agent_turns[i+1]["start"]
        if silence_end - silence_start > long_silence_threshold:
            caller_reactions = [ct for ct in caller_turns if silence_start < ct["start"] < silence_end]
            if caller_reactions:
                silence_reactions.append(caller_reactions[0]["start"] - silence_start)
                
    if silence_reactions:
        features["silence_reaction_latency_mean"] = float(np.mean(silence_reactions))
        
    # 4. Backchanneling (calibrado a 1.0s)
    backchannel_threshold = 1.0
    backchannels = 0
    for ct in caller_turns:
        if (ct["end"] - ct["start"]) <= backchannel_threshold:
            for at in agent_turns:
                if at["start"] <= ct["start"] <= at["end"]:
                    backchannels += 1
                    break 
                    
    features["backchannel_rate"] = float(backchannels / (total_duration / 60))
    
    return features