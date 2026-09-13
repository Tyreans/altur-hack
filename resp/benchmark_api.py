import os
import time
import base64
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/detect"
DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def run_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"No se encontró {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    # Si tienes muchos, puedes filtrar para evaluar solo el split de validación:
    # df = df[df['split'] == 'val'] 
    
    total_requests = 0
    correct_predictions = 0
    latencies = []
    fase_count = {"fast": 0, "heavy": 0, "fast_fallback": 0}

    print(f"Iniciando evaluación de {len(df)} audios contra {API_URL}...\n")

    for index, row in df.iterrows():
        anon_id = row['anon_id']
        true_label = str(row['label']).strip().lower() # 'human' o 'synthetic'
        wav_path = os.path.join(DATA_DIR, f"{anon_id}.wav")

        if not os.path.exists(wav_path):
            continue

        # 1. Preparar Payload
        with open(wav_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {"audio_base64": audio_b64}

        # 2. Medir tiempo de la petición exacta
        start_time = time.perf_counter()
        
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Error procesando {anon_id}: {e}")
            continue
            
        end_time = time.perf_counter()
        
        # 3. Procesar resultados
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        data = response.json()
        pred_is_synthetic = data.get("is_synthetic")
        fase_utilizada = data.get("phase", "desconocida")
        # Extraemos la métrica de confianza del JSON
        confianza = data.get("confidence", 0.0) 
        
        # Contabilizar por qué fase pasó 
        if fase_utilizada in fase_count:
            fase_count[fase_utilizada] += 1
            
        # Evaluar acierto
        pred_label = "synthetic" if pred_is_synthetic else "human"
        is_correct = (pred_label == true_label)
        if is_correct:
            correct_predictions += 1

        total_requests += 1
        estado = "BIEN" if is_correct else "X"
        
        # Agregamos "Conf: {confianza:.2f}" al reporte visual
        print(f"[{total_requests}/{len(df)}] {anon_id} | Verdad: {true_label[:4]} | Pred: {pred_label[:4]} {estado} | Conf: {confianza:.2f} | Lat: {latency_ms:.1f} ms | Fase: {fase_utilizada}")

    # --- REPORTE FINAL ---
    if total_requests > 0:
        acc = (correct_predictions / total_requests) * 100
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print("\n" + "="*40)
        print(" REPORTE FINAL DEL BENCHMARK ")
        print("="*40)
        print(f"Total procesados : {total_requests}")
        print(f"Precisión (Acc)  : {acc:.2f}%")
        print(f"Latencia Promedio: {avg_latency:.2f} ms")
        print(f"Latencia Mínima  : {min_latency:.2f} ms")
        print(f"Latencia Máxima  : {max_latency:.2f} ms")
        print(f"Resolución       : {fase_count}")
        print("="*40)

if __name__ == "__main__":
    run_benchmark()