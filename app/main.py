# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import joblib
import os

# Globals populated on startup
MODEL = None
CLASS_NAMES: List[str] = []
FEATURE_NAMES: List[str] = []
METRICS = {}

class IrisInput(BaseModel):
    """Input schema for a single Iris sample."""
    sepal_length: float = Field(..., gt=0, description="Sepal length (cm)")
    sepal_width: float = Field(..., gt=0, description="Sepal width (cm)")
    petal_length: float = Field(..., gt=0, description="Petal length (cm)")
    petal_width: float = Field(..., gt=0, description="Petal width (cm)")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

app = FastAPI(
    title="ML Model API",
    description="API for Iris species prediction using a trained scikit-learn model",
    version="1.0.0",
)

@app.on_event("startup")
def load_model():
    """Load serialized pipeline + metadata on app startup."""
    global MODEL, CLASS_NAMES, FEATURE_NAMES, METRICS

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file not found at {model_path}. Did you run 'python train.py' first?"
        )

    bundle = joblib.load(model_path)
    MODEL = bundle["model"]
    CLASS_NAMES = [str(x) for x in bundle.get("class_names", [])]
    FEATURE_NAMES = [str(x) for x in bundle.get("feature_names", [])]
    METRICS = bundle.get("metrics", {})

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "message": "API is running"
    }

@app.get("/model-info")
def model_info():
    return {
        "model_type": type(MODEL).__name__ if MODEL else None,
        "problem_type": "classification",
        "features": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "metrics": METRICS,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: IrisInput):
    try:
        if MODEL is None:
            raise RuntimeError("Model not loaded")

        # Convert to 2D array: [[sepal_length, sepal_width, petal_length, petal_width]]
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width,
        ]])

        # Predict class index + probabilities
        probs = MODEL.predict_proba(features)[0]
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx] if CLASS_NAMES else str(pred_idx)
        confidence = float(probs[pred_idx])

        return PredictionOutput(prediction=pred_name, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# (Optional) Batch prediction endpoint
class BatchInput(BaseModel):
    items: List[IrisInput]

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]

@app.post("/predict-batch", response_model=BatchOutput)
def predict_batch(batch: BatchInput):
    try:
        if MODEL is None:
            raise RuntimeError("Model not loaded")
        if not batch.items:
            raise ValueError("'items' cannot be empty")

        X = np.array([[
            it.sepal_length,
            it.sepal_width,
            it.petal_length,
            it.petal_width,
        ] for it in batch.items])

        probs = MODEL.predict_proba(X)
        outputs: List[PredictionOutput] = []
        for row in probs:
            idx = int(np.argmax(row))
            name = CLASS_NAMES[idx] if CLASS_NAMES else str(idx)
            conf = float(row[idx])
            outputs.append(PredictionOutput(prediction=name, confidence=conf))

        return BatchOutput(predictions=outputs)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
