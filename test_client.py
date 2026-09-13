import base64
import requests
import json

# Ruta de uno de los audios que validamos
audio_path = r"data\call_0cf2f4a71328.wav"

# 1. Leer los bytes crudos y codificarlos en Base64
with open(audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# 2. Armar el payload que espera tu schema DetectRequest
payload = {
    "audio_base64": audio_b64
}

# 3. Disparar la petición POST al servidor local
url = "http://127.0.0.1:8000/detect"
print(f"Enviando petición a {url}...")

try:
    response = requests.post(url, json=payload)
    
    # 4. Imprimir el veredicto
    if response.status_code == 200:
        print("¡Respuesta del servidor:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error HTTP {response.status_code}: {response.text}")
except requests.exceptions.ConnectionError:
    print("Error: No se pudo conectar. ¿El servidor Uvicorn está corriendo?")