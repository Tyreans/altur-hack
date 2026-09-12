"""
Script de entrenamiento para el clasificador de voz sintética.
Combina features acústicos y conductuales, entrena un Random Forest
y guarda el modelo en disco.
"""
import os
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

from .acoustic_features import extract_acoustic_features
from .behavior_features import extract_behavior_features

# Asumimos que se ejecuta desde la raíz del proyecto (altur-hack/)
DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def extract_features_for_row(row):
    anon_id = row['anon_id']
    wav_path = os.path.join(DATA_DIR, f"{anon_id}.wav")
    
    if not os.path.exists(wav_path):
        print(f"Warning: No se encontró {wav_path}")
        return None
        
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim != 2 or audio.shape[1] != 2:
            print(f"Warning: {wav_path} no es estéreo. Ignorando.")
            return None
            
        caller_audio = np.ascontiguousarray(audio[:, 0])
        agent_audio = np.ascontiguousarray(audio[:, 1])
        
        # 1. Acústicos (sólo caller)
        acoustic_feats = extract_acoustic_features(caller_audio, sr)
        
        # 2. Conductuales (ambos)
        behavior_feats = extract_behavior_features(caller_audio, agent_audio, sr)
        
        # Combinar diccionarios
        combined = {**acoustic_feats, **behavior_feats}
        combined["anon_id"] = anon_id
        return combined
    except Exception as e:
        print(f"Error procesando {wav_path}: {e}")
        return None

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: No se encontró {MANIFEST_PATH}")
        return

    print(f"Cargando manifest desde {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH)
    
    print(f"Total registros en el CSV: {len(df)}")
    print("Balance de clases en el CSV:")
    print(df['label'].value_counts())
    
    # Extraer features
    print("\nExtrayendo features de los audios (esto puede tomar un rato)...")
    features_list = []
    
    # Iterar con algo de progreso visual simple
    for i, row in df.iterrows():
        if i % 10 == 0:
            print(f"Procesando {i}/{len(df)}...")
            
        feats = extract_features_for_row(row)
        if feats is not None:
            features_list.append(feats)
            
    if not features_list:
        print("No se extrajeron features. Abortando. Verifica si los .wav están en data/")
        return
        
    features_df = pd.DataFrame(features_list)
    print(f"Features extraídos exitosamente para {len(features_df)} audios.")
    
    # Unir labels y split al DataFrame de features usando anon_id
    dataset = pd.merge(features_df, df[['anon_id', 'label', 'split']], on='anon_id')
    
    # Mapear clase: human=0, synthetic=1
    dataset['target'] = dataset['label'].apply(lambda x: 1 if str(x).strip().lower() == 'synthetic' else 0)
    
    # Separar train y val según la columna split explícita que viene en el CSV
    train_df = dataset[dataset['split'] == 'train']
    val_df = dataset[dataset['split'] == 'val']
    
    print(f"\nTamaño conjunto de entrenamiento (train): {len(train_df)}")
    print(f"Tamaño conjunto de validación (val): {len(val_df)}")
    
    if len(train_df) == 0 or len(val_df) == 0:
        print("Error: Uno de los splits (train o val) está vacío. Abortando.")
        return
    
    # Seleccionar solo las columnas numéricas generadas por los extractores
    feature_cols = [c for c in train_df.columns if c not in ['anon_id', 'label', 'split', 'target']]
    
    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df['target']
    
    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df['target']
    
    # Entrenar Random Forest
    print("\nEntrenando RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Evaluar
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    
    print("\n" + "="*40)
    print("      RESULTADOS EN VALIDACIÓN")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC:  {auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_val, y_pred, target_names=['Human (0)', 'Synthetic (1)']))
    
    # Importancia de features para la explicación ("Technical depth")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 15 Features más discriminativos:")
    for f in range(min(15, len(feature_cols))):
        print(f"{f+1:2d}. {feature_cols[indices[f]]:<30} ({importances[indices[f]]:.4f})")
        
    # Guardar modelo
    model_path = "model.joblib"
    # Guardamos también las columnas para asegurar el orden durante la inferencia en heavy_model.py
    joblib.dump({"model": clf, "feature_cols": feature_cols}, model_path)
    print(f"\nModelo entrenado y guardado en {model_path}!")

if __name__ == "__main__":
    main()
