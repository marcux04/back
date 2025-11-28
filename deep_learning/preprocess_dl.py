
import os
import joblib
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_preprocessor(df: pd.DataFrame, categorical_cols: list, numeric_cols: list):
    """
    Construye un ColumnTransformer que:
      - one-hot encodes categorical_cols (handle_unknown='ignore')
      - scales numeric_cols (StandardScaler)
    """
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_cols),
            ("num", num_pipe, numeric_cols)
        ],
        remainder="drop"
    )
    return preprocessor

def save_preprocessor(preproc, features: list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    preproc_path = os.path.join(save_dir, "preprocessor.pkl")
    features_path = os.path.join(save_dir, "features.json")
    joblib.dump(preproc, preproc_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f)
    return preproc_path, features_path

def load_preprocessor(save_dir: str):
    preproc_path = os.path.join(save_dir, "preprocessor.pkl")
    features_path = os.path.join(save_dir, "features.json")
    preproc = joblib.load(preproc_path)
    import json
    with open(features_path, "r", encoding="utf-8") as f:
        features = json.load(f)
    return preproc, features
