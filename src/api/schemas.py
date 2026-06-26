from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):

    status: str

    device: str

    model_loaded: bool


class PredictResponse(BaseModel):

    risk_score: float

    risk_level: str

    document_type: str

    confidence: float

    recommendation: str

    high_risk_clauses: list

    source: str | None = None

    file_type: str | None = None

    filename: str | None = None

    page_count: int | None = None

    ocr_engine: str | None = None