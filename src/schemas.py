"""
schemas.py — Pydantic Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictResponse(BaseModel):
    label: str = Field(..., example="FAKE")
    confidence: float = Field(..., example=0.97)
    probabilities: Dict[str, float] = Field(
        ..., example={"FAKE": 0.97, "REAL": 0.03}
    )
    latency_ms: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "label": "FAKE",
                "confidence": 0.97,
                "probabilities": {"FAKE": 0.97, "REAL": 0.03},
                "latency_ms": 18.4
            }
        }
