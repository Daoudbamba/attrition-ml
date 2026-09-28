import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.data import FEATURES
from src.schema import EmployeeFeatures, PredictionResponse

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "attrition_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

model_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        from src.train import run

        run()
    model_state["pipeline"] = joblib.load(MODEL_PATH)
    yield


app = FastAPI(title="Prédiction d'attrition — API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

web_dir = Path(__file__).resolve().parent.parent / "web"
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=web_dir, html=True), name="web")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Modèle pas encore entraîné")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/api/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures):
    pipeline = model_state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    row = pd.DataFrame([employee.model_dump()])[FEATURES]
    probability = float(pipeline.predict_proba(row)[0, 1])
    return PredictionResponse(attrition_probability=probability, attrition_predite=probability >= 0.5)
