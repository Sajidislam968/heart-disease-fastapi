"""FastAPI application for a saved heart-disease classifier."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartFeatures, PredictionResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "model" / "heart_model.joblib"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", ROOT_DIR / "model" / "metadata.json"))
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
DEFAULT_METADATA: dict[str, Any] = {
    "model_type": "LogisticRegression (StandardScaler + LogisticRegression pipeline)",
    "features": FEATURES,
    "target": {"column": "target", "0": "absence", "1": "presence"},
    "dataset": "Kaggle johnsmith88/heart-disease-dataset (heart.csv)",
}

MODEL: Any | None = None
MODEL_MTIME: float | None = None
_METADATA_CACHE: dict[str, Any] | None = None

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "A Docker-ready educational demo that predicts the dataset's binary heart-disease label. "
        "It is not intended for clinical use."
    ),
    version="1.0.0",
)


def get_model() -> Any | None:
    """Load the joblib model once, and reload it if the file changes."""
    global MODEL, MODEL_MTIME
    if not MODEL_PATH.is_file():
        return None
    mtime = MODEL_PATH.stat().st_mtime
    if MODEL is None or MODEL_MTIME != mtime:
        try:
            MODEL = joblib.load(MODEL_PATH)
            MODEL_MTIME = mtime
            logger.info("Loaded prediction model from %s", MODEL_PATH)
        except Exception:
            logger.exception("Could not load model artifact at %s", MODEL_PATH)
            MODEL = None
            MODEL_MTIME = None
    return MODEL


def get_metadata() -> dict[str, Any]:
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE
    if METADATA_PATH.is_file():
        try:
            _METADATA_CACHE = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read model metadata; using defaults")
    if _METADATA_CACHE is None:
        _METADATA_CACHE = DEFAULT_METADATA.copy()
    return _METADATA_CACHE


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    return {
        "message": "Heart Disease Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["General"])
def health() -> dict[str, str | bool]:
    model_loaded = get_model() is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.get("/info", tags=["General"])
def info() -> dict[str, Any]:
    metadata = get_metadata()
    return {
        "name": "Heart Disease Prediction API",
        "model_type": metadata.get("model_type", DEFAULT_METADATA["model_type"]),
        "features": metadata.get("features", FEATURES),
        "feature_count": len(metadata.get("features", FEATURES)),
        "target": metadata.get("target", DEFAULT_METADATA["target"]),
        "dataset": metadata.get("dataset", DEFAULT_METADATA["dataset"]),
        "model_loaded": get_model() is not None,
        "disclaimer": "Educational demo only; predictions are not medical advice.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: HeartFeatures) -> PredictionResponse:
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model is not available. Run `python scripts/train_model.py` "
                "or build the Docker image, which trains the model during build."
            ),
        )

    feature_frame = pd.DataFrame([features.to_feature_dict()], columns=FEATURES)
    try:
        prediction = int(model.predict(feature_frame)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(feature_frame)[0][1])
        else:
            probability = float(prediction)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed for the supplied input.") from exc

    positive = prediction == 1
    metadata = get_metadata()
    return PredictionResponse(
        heart_disease=positive,
        prediction="presence" if positive else "absence",
        probability=round(probability, 6),
        model=metadata.get("model_type", DEFAULT_METADATA["model_type"]),
    )
