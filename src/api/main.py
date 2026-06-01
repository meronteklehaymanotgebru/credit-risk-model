"""FastAPI service that serves the registered credit risk model."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import FeatureInput, PredictionResponse

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Model loaded at startup
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "1"

try:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.sklearn.load_model(model_uri)
    app.state.model_version = f"{MODEL_NAME} v{MODEL_VERSION}"
except Exception as e:
    print(f"ERROR: Could not load model: {e}")
    model = None
    app.state.model_version = "none"

@app.get("/")
def root():
    return {"status": "ok", "model": app.state.model_version}

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: FeatureInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to NumPy array and reshape to (1, n_features)
    X = np.array(input_data.data).reshape(1, -1)

    proba = model.predict_proba(X)[0, 1]   # probability of class 1 (high risk)
    category = "high risk" if proba >= 0.5 else "low risk"

    return PredictionResponse(
        risk_probability=float(proba),
        risk_category=category,
        model_version=app.state.model_version
    )