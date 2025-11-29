from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from deep_learning.predict_dl import predict_from_dict, load_model_and_preproc
from deep_learning.utils_dl import save_prediction_to_mongo, save_training_log

load_dotenv()
router = APIRouter(prefix="/dl", tags=["deep_learning"])

MODELS_DIR = os.getenv("DL_MODELS_DIR", "deep_learning/models")

class InputData(BaseModel):
    # Reproduce las mismas columnas que usas en tu app (simplificado ejemplo)
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@router.post("/predict")
def predict(input_data: InputData):
    try:
        data = input_data.dict()
        print(data)
        res = predict_from_dict(data, MODELS_DIR)
        # guardar en mongo
        save_prediction_to_mongo({"input": data, "result": res})
        return res
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
def info():
    # devuelve info del modelo (si existe)
    try:
        model, preproc, features = load_model_and_preproc(MODELS_DIR)
        return {"status": "ok", "features_count": len(features), "features": features[:50]}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo DL no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain")
def retrain():
    """
    Endpoint simple que ejecuta el script train_dl.py que ya está en el repo.
    En producción quizá prefieras lanzar un job asíncrono o un pipeline.
    """
    import subprocess, sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script_path = os.path.join(project_root, "deep_learning", "train_dl.py")
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"No se encontró {script_path}")
    try:
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True, cwd=project_root)
        # guardar log en mongo
        save_training_log({"stdout": result.stdout, "stderr": result.stderr})
        return {"message": "Retrain started", "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        save_training_log({"error": e.stderr})
        raise HTTPException(status_code=500, detail=e.stderr)