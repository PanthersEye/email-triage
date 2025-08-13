from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class HealthResponse(BaseModel):
    status: str = "ok"

class TrainSynthRequest(BaseModel):
    n_per_cat: int = Field(12, ge=1, le=200, description="Samples to generate per category.")

class TrainResponse(BaseModel):
    samples: int
    classes: List[str]
    accuracy: float

class PredictRequest(BaseModel):
    subject: str = Field(..., example="Login issue")
    body: str = Field(..., example="I can’t access my dashboard despite the right credentials.")

class PredictResponse(BaseModel):
    category: str
    probabilities: Dict[str, float]
    priority_score: float
    priority_label: str

class BatchEmail(BaseModel):
    subject: str
    body: str

class BatchPredictRequest(BaseModel):
    items: List[BatchEmail]

class BatchPredictItem(BaseModel):
    category: str
    probabilities: Dict[str, float]
    priority_score: float
    priority_label: str

class BatchPredictResponse(BaseModel):
    results: List[BatchPredictItem]

class ReplyRequest(BaseModel):
    subject: str
    body: str
    guidance: Optional[str] = "Generate a concise, professional reply."

class ReplyResponse(BaseModel):
    draft: str
