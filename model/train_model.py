# back/model/train_model.py
import os
import json
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import joblib
from pymongo import MongoClient
from dotenv import load_dotenv
import sklearn

# --- Cargar variables de entorno ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bank_marketing_db")

# --- Conectar a MongoDB (si falla, salimos con mensaje) ---
if not MONGO_URI:
    print("[ERROR] MONGO_URI no está definido en variables de entorno. Abortando.")
    raise SystemExit(1)

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    results_collection = db["results"]
    print("[OK] Conexión a MongoDB establecida.")
except Exception as e:
    print(f"[ERROR] Error al conectar a MongoDB: {e}")
    raise

# --- Rutas ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "bank.csv")
OUT_MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(OUT_MODEL_DIR, exist_ok=True)

print("[INFO] Cargando dataset desde:", DATA_PATH)
try:
    df = pd.read_csv(DATA_PATH, sep=";")
except FileNotFoundError:
    print(f"[ERROR] No se encontró el archivo {DATA_PATH}. Asegúrate que exista.")
    raise
except Exception as e:
    print(f"[ERROR] Error leyendo dataset: {e}")
    raise

# --- Preprocesamiento ---
print("[INFO] Preprocesando datos...")
if "y" not in df.columns:
    raise SystemExit("[ERROR] El dataset no contiene la columna 'y' necesaria.")

df["y"] = df["y"].map({"yes": 1, "no": 0})

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = df[col].fillna("unknown")
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Separar X,y ---
X = df.drop(columns=["y"])
y = df["y"]

# --- Split ---
print("[INFO] Dividiendo dataset (train/test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Entrenamiento ---
print("[INFO] Entrenando RandomForest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predicciones y métricas ---
print("[INFO] Evaluando modelo...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Métricas con la estructura antigua que funciona con tu Streamlit
metrics = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# --- Guardado de artefactos ---
# Legacy filenames (compatibilidad con front/back existentes)
legacy_model_path = os.path.join(OUT_MODEL_DIR, "model.pkl")
legacy_enc_path = os.path.join(OUT_MODEL_DIR, "label_encoders.pkl")

# Versioned artifact names
timestamp_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
skl_ver = sklearn.__version__
np_ver = np.__version__
model_tag = f"rf_{timestamp_tag}_skl{skl_ver.replace('.', '-')}"
model_file = os.path.join(OUT_MODEL_DIR, f"{model_tag}.pkl")
enc_file = os.path.join(OUT_MODEL_DIR, f"{model_tag}_encoders.pkl")
meta_file = os.path.join(OUT_MODEL_DIR, f"{model_tag}_metadata.json")

print("[INFO] Guardando modelos y encoders...")
joblib.dump(model, model_file)
joblib.dump(label_encoders, enc_file)

# también mantener los archivos "current/legacy" para compatibilidad
joblib.dump(model, legacy_model_path)
joblib.dump(label_encoders, legacy_enc_path)

# Guardar metadata/version info
metadata = {
    "model_tag": model_tag,
    "sklearn_version": skl_ver,
    "numpy_version": np_ver,
    "timestamp": timestamp_tag,
    "features": list(X.columns),
    "metrics": metrics
}
with open(meta_file, "w", encoding="utf8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# actualizar "current_*" (opcional)
current_model = os.path.join(OUT_MODEL_DIR, "current_model.pkl")
current_enc = os.path.join(OUT_MODEL_DIR, "current_label_encoders.pkl")
current_meta = os.path.join(OUT_MODEL_DIR, "current_metadata.json")

shutil.copyfile(model_file, current_model)
shutil.copyfile(enc_file, current_enc)
shutil.copyfile(meta_file, current_meta)

print(f"[OK] Modelo guardado: {model_file}")
print(f"[OK] Encoders guardados: {enc_file}")
print(f"[OK] Metadata guardada: {meta_file}")
print(f"[OK] Legacy model guardado: {legacy_model_path}")
print(f"[OK] Legacy encoders guardado: {legacy_enc_path}")

# --- Guardar métricas en MongoDB (manteniendo la estructura antigua) ---
try:
    # si deseas limpiar métricas anteriores (comportamiento antiguo), descomenta:
    # results_collection.delete_many({})
    results_collection.insert_one(metrics)
    print("[OK] Métricas guardadas en MongoDB (estructura antigua).")
except Exception as e:
    print(f"[ERROR] Error guardando métricas en MongoDB: {e}")

# --- Resumen por consola ---
print("\n[RESULTADOS DEL MODELO]")
for k, v in metrics.items():
    if isinstance(v, (float, int)):
        print(f"{k}: {v:.4f}")
    else:
        print(f"{k}: {v}")

print("\n[FIN]")
