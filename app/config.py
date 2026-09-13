# Umbrales de la Fase 1 (heurísticas rápidas).
# Si la confianza de "es sintético" cae FUERA de este rango medio,
# respondemos directo sin pasar por el modelo pesado.
FAST_HIGH_THRESHOLD = 0.95  # >= esto -> "sintético", responde ya
FAST_LOW_THRESHOLD = 0.05   # <= esto -> "humano", responde ya

# Timeout máximo (segundos) para esperar al modelo pesado (Fase 2).
HEAVY_MODEL_TIMEOUT = 30.0

# Nota para Paola: si tienen tiempo de correr el dataset .json contra
# fast_checks.py, estos umbrales son lo primero que hay que recalibrar
# con datos reales en vez de dejarlos a ojo.


import os

# Limit concurrent heavy processing to prevent CPU thrashing
# For a 16-thread Ryzen, processing 7-8 concurrent audios on a single thread each is optimal.
MAX_CONCURRENT_HEAVY = max(1, (os.cpu_count() or 4) - 1)
HEAVY_MODEL_TIMEOUT = 30.0