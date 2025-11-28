# back/api/router_metrics.py
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "bank_marketing_db")

router = APIRouter()

def get_db():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]

@router.get("/metrics/normalized")
def metrics_normalized():
    db = get_db()
    doc = db["results"].find_one({}, {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="No metrics found")
    # si el documento tiene la forma { "metrics": {...} }
    if "metrics" in doc and isinstance(doc["metrics"], dict):
        return {"metrics": doc["metrics"], "model_tag": doc.get("model_tag"), "timestamp": doc.get("timestamp")}
    # fallback a formato plano
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"]
    metrics = {k: doc.get(k) for k in keys}
    return {"metrics": metrics, "model_tag": doc.get("model_tag"), "timestamp": doc.get("timestamp")}
