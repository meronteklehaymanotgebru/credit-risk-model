"""Request / response schemas for the credit risk API."""

from pydantic import BaseModel
from typing import List

class FeatureInput(BaseModel):
    """Single prediction input matching the model features."""
    data: List[float]   # 32 numerical values (after encoding and scaling)

class PredictionResponse(BaseModel):
    """Output returned to the client."""
    risk_probability: float
    risk_category: str
    model_version: str