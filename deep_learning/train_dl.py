
"""
Script de entrenamiento para MLP.
Uso (desde la raíz del backend):
python -m deep_learning.train_dl --data_path ./data/bank.csv --save_dir deep_learning/models
"""
import os
import argparse
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from .preprocess_dl import build_preprocessor, save_preprocessor
from .model_dl import build_mlp
from pymongo import MongoClient
from dotenv import load_dotenv
import tensorflow as tf 

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bank_marketing_db")

def default_feature_lists(df):
    # heurístico: columnas categóricas vs numéricas (excluir target 'y' si existe)
    cols = df.columns.tolist()
    if "y" in cols:
        cols.remove("y")
    categorical = df.select_dtypes(include=["object"]).columns.tolist()
    numeric = [c for c in cols if c not in categorical]
    return categorical, numeric

def save_metrics_to_mongo(metrics: dict, save_collection="dl_training_logs"):
    if MONGO_URI:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        db[save_collection].insert_one(metrics)
    else:
        print("MONGO_URI no configurado — métricas no guardadas en MongoDB")

def main(args):
    df = pd.read_csv(args.data_path, sep=args.sep)
    # assume target column named 'y' with values 'yes'/'no'
    if "y" not in df.columns:
        raise RuntimeError("El dataset debe contener columna 'y' con 'yes'/'no'")

    df = df.dropna(subset=["y"])  # simple limpieza
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    categorical_cols, numeric_cols = default_feature_lists(df)

    X = df.drop(columns=["y"])
    y = df["y"]

    preproc = build_preprocessor(X, categorical_cols, numeric_cols)
    X_transformed = preproc.fit_transform(X)  # numpy array
    feature_dim = X_transformed.shape[1]

    model = build_mlp(input_dim=feature_dim,
                      hidden_layers=args.hidden_layers,
                      dropout=args.dropout)

    X_train, X_val, y_train, y_val = train_test_split(X_transformed, y.values,
                                                      test_size=args.test_size, random_state=42, stratify=y)

    callbacks = [
        # early stopping
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir, "mlp_model.h5"), save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    # evaluar
    y_pred_proba = model.predict(X_val).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_pred_proba)),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "train_args": vars(args)
    }

    # guardar modelo (ya guardado por ModelCheckpoint) y preprocessor + feature list
    os.makedirs(args.save_dir, exist_ok=True)
    preproc_path, features_path = save_preprocessor(preproc, (categorical_cols + numeric_cols), args.save_dir)

    # guardar historia
    hist_path = os.path.join(args.save_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

    # guardar métricas a MongoDB
    metrics_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": metrics
    }
    save_metrics_to_mongo(metrics_record)

    print("Entrenamiento finalizado. Métricas:", metrics)
    print("Model saved to:", os.path.join(args.save_dir, "mlp_model.h5"))
    print("Preprocessor saved to:", preproc_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/bank.csv")
    parser.add_argument("--save_dir", type=str, default="./deep_learning/models")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--hidden_layers", type=lambda s: [int(x) for x in s.split(",")], default=[128,64])
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
