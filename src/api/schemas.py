from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):

    status: str

    device: str

    model_loaded: bool


class ClauseDetail(BaseModel):
    """Individual detected clause with probability."""

    clause: str

    probability: float


class PredictResponse(BaseModel):

    risk_score: float

    risk_level: str

    document_type: str

    confidence: float

    recommendation: str

    high_risk_clauses: List[ClauseDetail]

    source: str | None = None

    file_type: str | None = None

    filename: str | None = None

    page_count: int | None = None

    ocr_engine: str | None = None

    ocr_text: str | None = None

    execution_time_seconds: float | None = None


class BatchPredictResponse(BaseModel):
    """Response for batch prediction endpoint."""

    results: List[PredictResponse]

    total_files: int

    total_execution_time_seconds: float


class ModelInfoResponse(BaseModel):
    """Response for model information endpoint."""

    model_name: str

    vision_backbone: str

    text_backbone: str

    document_classes: int

    clause_classes: int

    device: str

    total_parameters: int

    trainable_parameters: int


class VersionResponse(BaseModel):
    """Response for version endpoint."""

    api_version: str

    python_version: str

    pytorch_version: str

    cuda_available: bool

    cuda_version: str | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str

    detail: str | None = None

    status_code: int = 500