import os
import joblib
import numpy as np
from tensorflow import keras
from .preprocess_dl import load_preprocessor

def load_model_and_preproc(models_dir: str):
    model_path = os.path.join(models_dir, "mlp_model.h5")
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")
    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        raise FileNotFoundError("Modelo o preprocessor no encontrado. Ejecuta train_dl.py")
    model = keras.models.load_model(model_path)
    preproc, features = load_preprocessor(models_dir)
    return model, preproc, features

def predict_from_dict(input_dict: dict, models_dir: str):
    """
    input_dict: diccionario con las features crudas (nombres tal como en features.json)
    """
    model, preproc, features = load_model_and_preproc(models_dir)
    import pandas as pd
    df = pd.DataFrame([input_dict])
    # Ensure columns present
    # Preprocessor expects in order used in training: we'll attempt to select these columns
    # features may list categorical then numeric (como guardamos)
    missing = [c for c in features if c not in df.columns]
    for m in missing:
        df[m] = 0  # valor neutro
    X_trans = preproc.transform(df[features])
    proba = float(model.predict(X_trans).ravel()[0])
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability_yes": proba}