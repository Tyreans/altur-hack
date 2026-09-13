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

DATA_DIR = "data"
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")

def extract_features_for_row(row):
    anon_id = row['anon_id']
    wav_path = os.path.join(DATA_DIR, "audios", f"{anon_id}.wav")
    
    if not os.path.exists(wav_path):
        return None
        
    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim != 2 or audio.shape[1] != 2:
            return None
            
        caller_audio = np.ascontiguousarray(audio[:, 0])
        agent_audio = np.ascontiguousarray(audio[:, 1])
        
        acoustic_feats = extract_acoustic_features(caller_audio, sr)
        behavior_feats = extract_behavior_features(caller_audio, agent_audio, sr)
        
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

    df = pd.read_csv(MANIFEST_PATH)

   
    features_list = []
    
    for i, row in df.iterrows():
        feats = extract_features_for_row(row)
        if feats is not None:
            features_list.append(feats)
            
    features_df = pd.DataFrame(features_list)
    dataset = pd.merge(features_df, df[['anon_id', 'label', 'split']], on='anon_id')
    dataset['target'] = dataset['label'].apply(lambda x: 1 if str(x).strip().lower() == 'synthetic' else 0)
    
    train_df = dataset[dataset['split'] == 'train'].copy()
    val_df = dataset[dataset['split'] == 'val'].copy()
    
    feature_cols = [c for c in train_df.columns if c not in ['anon_id', 'label', 'split', 'target']]
    
    # 1. Diagnóstico de NaNs
    nan_counts = train_df[feature_cols].isna().sum()
    print("\n--- Diagnóstico de NaNs (Train) ---")
    for col in feature_cols:
        if nan_counts[col] > 0:
            nan_human = train_df[(train_df['target'] == 0)][col].isna().sum()
            nan_synth = train_df[(train_df['target'] == 1)][col].isna().sum()
            print(f"{col}: {nan_counts[col]} total (Humanos: {nan_human}, IA: {nan_synth})")

    # Guardar AUC con Data Leakage para comparación
    clf_leakage = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf_leakage.fit(train_df[feature_cols].fillna(0.0), train_df['target'])
    auc_leakage = roc_auc_score(val_df['target'], clf_leakage.predict_proba(val_df[feature_cols].fillna(0.0))[:, 1])

    # 2. Imputación correcta (Solo Mediana de Train + _was_missing)
    medians = train_df[feature_cols].median()
    
    for col in feature_cols:
        if nan_counts[col] > 0:
            train_df[f"{col}_was_missing"] = train_df[col].isna().astype(float)
            val_df[f"{col}_was_missing"] = val_df[col].isna().astype(float)
            
            train_df[col] = train_df[col].fillna(medians[col])
            val_df[col] = val_df[col].fillna(medians[col])
            
    final_feature_cols = [c for c in train_df.columns if c not in ['anon_id', 'label', 'split', 'target']]
    
    X_train = train_df[final_feature_cols]
    y_train = train_df['target']
    X_val = val_df[final_feature_cols]
    y_val = val_df['target']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auc_clean = roc_auc_score(y_val, y_prob)
    
    print("\n--- Reporte de Fix de Leakage ---")
    print(f"AUC con fillna(0.0) global : {auc_leakage:.4f}")
    print(f"AUC con mediana train+flags: {auc_clean:.4f}")
    
    print("\nTop 15 Features más discriminativos (Post-fix):")
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for f in range(min(15, len(final_feature_cols))):
        print(f"{f+1:2d}. {final_feature_cols[indices[f]]:<30} ({importances[indices[f]]:.4f})")
        
    model_path = "model.joblib"
    joblib.dump({
        "model": clf, 
        "feature_cols": final_feature_cols,
        "medians": medians
    }, model_path)
    print(f"\nModelo entrenado y guardado en {model_path}!")

if __name__ == "__main__":
    main()