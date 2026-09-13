"""
import os
import time
import base64
import requests
import pandas as pd
import soundfile as sf
import numpy as np

API_URL = "http://127.0.0.1:8000/detect"
DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def run_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"No se encontró {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    lista_audios_prueba = [
        "call_d3e854cbb77d", "call_d437831f4543", "call_d4518fcb700a", "call_d557fa43925b",
        "call_d5d00276dd35", "call_d60330be1b1f", "call_d72a7cb67eb0", "call_d80d9f6f7884",
        "call_d88d3c23953c", "call_d88d8e15e430", "call_d89caa393417", "call_d9b9c625f406",
        "call_db5326bcb672", "call_dc1a55102190", "call_dcdbcc7d70e4", "call_ddeabfdb99d4",
        "call_ddf2da1910f1", "call_df3e278ab7fe", "call_df75973c8794", "call_e0a8703be347",
        "call_e0c1f661d9ca", "call_e106a61086bb", "call_e118b75d4937", "call_e2200941723b",
        "call_e2d538297f0c", "call_e3064936e37a", "call_e3084824935d", "call_e31bf946a8fa",
        "call_e329faf2b6a1", "call_e3f555297277", "call_e4d9c2e0947a", "call_e58669998b76",
        "call_e618b7b1fa5b", "call_e697765a4dfd", "call_e69e8ba551e4", "call_e6c59d28e6d7",
        "call_e9bd33cdb226", "call_eb1c9a5a2c08", "call_ec545493459e", "call_ec58505fc522",
        "call_ed020e30c242", "call_ed041799aed9", "call_ed99e3ea3f0e", "call_ee227bc9d7ac",
        "call_eed2c7389776", "call_f003741555a1", "call_f0ad1cf1aa6a", "call_f1372119b403",
        "call_f1b82a4a46f6", "call_f202d4ff9f96", "call_f23bfe7d0e9e", "call_f27db4ab1f5a",
        "call_f2ab1c408b36", "call_f331303c4fb8", "call_f5251fb3b1d9", "call_f68d3459fdc7",
        "call_f6ce5de7ac09", "call_f9fbabf70f7e", "call_fa26d1186721", "call_fa4d7f2897f4",
        "call_fa9e18dd83dd", "call_fb1405943fed", "call_fc8bda4b3374", "call_fce92bad5095",
        "call_fdc3a7dac09f", "call_fdc44f969ade", "call_fe828697f60d", "call_fe8e4fc3ef69",
        "call_feaca5a8cd14", "call_ff4042587962", "call_fff8cae57de7"
    ]
    
    df = df[df['anon_id'].isin(lista_audios_prueba)]
    
    total_requests = 0
    correct_predictions = 0
    latencies = []
    durations = []
    fase_count = {"fast": 0, "heavy": 0, "fast_fallback": 0}

    print(f"Iniciando evaluación de {len(df)} audios contra {API_URL}...\n")

    for index, row in df.iterrows():
        anon_id = row['anon_id']
        true_label = str(row['label']).strip().lower()
        wav_path = os.path.join(DATA_DIR, "audios", f"{anon_id}.wav")

        if not os.path.exists(wav_path):
            continue

        try:
            audio_info = sf.info(wav_path)
            durations.append(audio_info.duration)
        except Exception:
            continue

        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {"audio_base64": audio_b64}
        start_time = time.perf_counter()
        
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Error procesando {anon_id}: {e}")
            durations.pop()
            continue
            
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        data = response.json()
        pred_is_synthetic = data.get("is_synthetic")
        fase_utilizada = data.get("phase", "desconocida")
        raw_conf = data.get("confidence", 0.0) 
        
        certeza = raw_conf if pred_is_synthetic else (1.0 - raw_conf)
        porcentaje_certeza = certeza * 100
        
        if fase_utilizada in fase_count:
            fase_count[fase_utilizada] += 1
            
        pred_label = "synthetic" if pred_is_synthetic else "human"
        is_correct = (pred_label == true_label)
        if is_correct:
            correct_predictions += 1

        total_requests += 1
        estado = "BIEN" if is_correct else "❌"
        
        print(f"[{total_requests}/{len(df)}] {anon_id} | Verdad: {true_label[:4]} | Pred: {pred_label[:4]} {estado} | Certeza: {porcentaje_certeza:.1f}% | Lat: {latency_ms:.1f} ms | Fase: {fase_utilizada}")

    if total_requests > 0:
        acc = (correct_predictions / total_requests) * 100
        avg_latency = sum(latencies) / len(latencies)
        
        correlacion = 0.0
        if len(durations) > 1 and len(durations) == len(latencies):
            correlacion = np.corrcoef(durations, latencies)[0, 1]

        print("\n" + "="*40)
        print(" REPORTE FINAL DEL BENCHMARK ")
        print("="*40)
        print(f"Total procesados : {total_requests}")
        print(f"Precisión (Acc)  : {acc:.2f}%")
        print(f"Latencia Promedio: {avg_latency:.2f} ms")
        print(f"Resolución       : {fase_count}")
        print(f"Correlación (Duración vs Latencia): {correlacion:.4f}")
        print("="*40)

if __name__ == "__main__":
    run_benchmark()
    """

import os
import time
import base64
import requests
import pandas as pd
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://127.0.0.1:8000/detect"
DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

# Cambiar a 1 para medir la latencia real individual sin embotellamientos
MAX_WORKERS = 4


def process_single_audio(row, total_audios):
    anon_id = row['anon_id']
    true_label = str(row['label']).strip().lower()
    wav_path = os.path.join(DATA_DIR, "audios", f"{anon_id}.wav")

    result = {
        "anon_id": anon_id,
        "true_label": true_label,
        "status": "failed",
        "latency_ms": 0.0,
        "is_correct": False,
        "phase": "unknown",
        "porcentaje_certeza": 0.0,
        "pred_label": "unknown"
    }

    if not os.path.exists(wav_path):
        result["error"] = "File not found"
        return result

    try:
        audio_info = sf.info(wav_path)
        result["duration"] = audio_info.duration
    except Exception as e:
        result["error"] = f"SF Error: {e}"
        return result

    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    payload = {"audio_base64": audio_b64}
    start_time = time.perf_counter()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        result["status"] = "timeout"
        return result
    except Exception as e:
        result["status"] = "http_error"
        result["error"] = str(e)
        return result
        
    end_time = time.perf_counter()
    result["latency_ms"] = (end_time - start_time) * 1000
    
    data = response.json()
    pred_is_synthetic = data.get("is_synthetic")
    raw_conf = data.get("confidence", 0.0)
    
    certeza = raw_conf if pred_is_synthetic else (1.0 - raw_conf)
    result["porcentaje_certeza"] = certeza * 100
    
    result["phase"] = data.get("phase", "desconocida")
    result["status"] = "success"
    
    result["pred_label"] = "synthetic" if pred_is_synthetic else "human"
    result["is_correct"] = (result["pred_label"] == true_label)

    return result


def run_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"No se encontró {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    
    lista_audios_prueba = [
        "call_d3e854cbb77d", "call_d437831f4543", "call_d4518fcb700a", "call_d557fa43925b",
        "call_d5d00276dd35", "call_d60330be1b1f", "call_d72a7cb67eb0", "call_d80d9f6f7884",
        "call_d88d3c23953c", "call_d88d8e15e430", "call_d89caa393417", "call_d9b9c625f406",
        "call_db5326bcb672", "call_dc1a55102190", "call_dcdbcc7d70e4", "call_ddeabfdb99d4",
        "call_ddf2da1910f1", "call_df3e278ab7fe", "call_df75973c8794", "call_e0a8703be347",
        "call_e0c1f661d9ca", "call_e106a61086bb", "call_e118b75d4937", "call_e2200941723b",
        "call_e2d538297f0c", "call_e3064936e37a", "call_e3084824935d", "call_e31bf946a8fa",
        "call_e329faf2b6a1", "call_e3f555297277", "call_e4d9c2e0947a", "call_e58669998b76",
        "call_e618b7b1fa5b", "call_e697765a4dfd", "call_e69e8ba551e4", "call_e6c59d28e6d7",
        "call_e9bd33cdb226", "call_eb1c9a5a2c08", "call_ec545493459e", "call_ec58505fc522",
        "call_ed020e30c242", "call_ed041799aed9", "call_ed99e3ea3f0e", "call_ee227bc9d7ac",
        "call_eed2c7389776", "call_f003741555a1", "call_f0ad1cf1aa6a", "call_f1372119b403",
        "call_f1b82a4a46f6", "call_f202d4ff9f96", "call_f23bfe7d0e9e", "call_f27db4ab1f5a",
        "call_f2ab1c408b36", "call_f331303c4fb8", "call_f5251fb3b1d9", "call_f68d3459fdc7",
        "call_f6ce5de7ac09", "call_f9fbabf70f7e", "call_fa26d1186721", "call_fa4d7f2897f4",
        "call_fa9e18dd83dd", "call_fb1405943fed", "call_fc8bda4b3374", "call_fce92bad5095",
        "call_fdc3a7dac09f", "call_fdc44f969ade", "call_fe828697f60d", "call_fe8e4fc3ef69",
        "call_feaca5a8cd14", "call_ff4042587962", "call_fff8cae57de7"
    ]
    
    df = df[df['anon_id'].isin(lista_audios_prueba)]
    total_audios = len(df)
    
    print(f"Iniciando evaluación concurrente de {total_audios} audios (Workers: {MAX_WORKERS})...\n")
    
    t_global_start = time.perf_counter()
    resultados = []
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_audio, row, total_audios) for _, row in df.iterrows()]
        
        # as_completed devuelve los resultados conforme van terminando, no en orden
        for future in as_completed(futures):
            res = future.result()
            resultados.append(res)
            completed_count += 1
            
            if res["status"] == "success":
                estado_str = "BIEN" if res["is_correct"] else "❌"
                print(f"[{completed_count}/{total_audios}] {res['anon_id']} | Verdad: {res['true_label'][:4]} | Pred: {res['pred_label'][:4]} {estado_str} | Certeza: {res['porcentaje_certeza']:.1f}% | Lat: {res['latency_ms']:.1f} ms | Fase: {res['phase']}")
            else:
                print(f"[{completed_count}/{total_audios}] {res['anon_id']} | ERROR: {res.get('error', res['status'])}")
            
    t_global_end = time.perf_counter()
    
    exitos = [r for r in resultados if r["status"] == "success"]
    
    if exitos:
        latencies = [r["latency_ms"] for r in exitos]
        correct_preds = sum(1 for r in exitos if r["is_correct"])
        
        acc = (correct_preds / len(exitos)) * 100
        avg_latency = np.mean(latencies)
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        print("\n" + "="*45)
        print("  REPORTE FINAL DEL BENCHMARK (EN MEMORIA) ")
        print("="*45)
        print(f"Total procesados : {len(exitos)}/{total_audios} (Fallos: {total_audios - len(exitos)})")
        print(f"Precisión (Acc)  : {acc:.2f}%")
        print(f"Tiempo Total     : {t_global_end - t_global_start:.2f} s")
        print("-" * 45)
        print(f"Latencia Avg     : {avg_latency:.2f} ms")
        print(f"Latencia Mediana : {median_latency:.2f} ms")
        print(f"Latencia P95     : {p95_latency:.2f} ms")
        print(f"Latencia Máxima  : {max_latency:.2f} ms")
        print("="*45)

if __name__ == "__main__":
    run_benchmark()