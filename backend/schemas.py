

from pydantic import BaseModel
from typing import List


class TopPrediction(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top3: List[TopPrediction]
    treatment_tip: str


class HealthResponse(BaseModel):
    status: str
    classes_loaded: int
