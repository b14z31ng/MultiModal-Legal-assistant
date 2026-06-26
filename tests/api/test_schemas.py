"""Tests for API Pydantic schemas."""

from __future__ import annotations

import pytest

from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    ClauseDetail,
    BatchPredictResponse,
    ModelInfoResponse,
    VersionResponse,
    ErrorResponse,
)


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_valid_health_response(self):
        """Should accept valid health data."""
        resp = HealthResponse(
            status="ok",
            device="cuda",
            model_loaded=True,
        )
        assert resp.status == "ok"
        assert resp.device == "cuda"
        assert resp.model_loaded is True


class TestClauseDetail:
    """Tests for ClauseDetail schema."""

    def test_valid_clause(self):
        """Should accept valid clause data."""
        clause = ClauseDetail(
            clause="Non-Compete",
            probability=0.85,
        )
        assert clause.clause == "Non-Compete"
        assert clause.probability == 0.85


class TestPredictResponse:
    """Tests for PredictResponse schema."""

    def test_valid_predict_response(self):
        """Should accept valid prediction data."""
        resp = PredictResponse(
            risk_score=67.5,
            risk_level="HIGH",
            document_type="letter",
            confidence=89.3,
            recommendation="Review needed.",
            high_risk_clauses=[
                ClauseDetail(clause="Test", probability=0.9),
            ],
        )
        assert resp.risk_score == 67.5
        assert resp.risk_level == "HIGH"
        assert len(resp.high_risk_clauses) == 1

    def test_optional_fields_default_none(self):
        """Optional fields should default to None."""
        resp = PredictResponse(
            risk_score=0,
            risk_level="LOW",
            document_type="unknown",
            confidence=0,
            recommendation="N/A",
            high_risk_clauses=[],
        )
        assert resp.source is None
        assert resp.file_type is None
        assert resp.filename is None
        assert resp.page_count is None
        assert resp.ocr_engine is None
        assert resp.ocr_text is None
        assert resp.execution_time_seconds is None


class TestBatchPredictResponse:
    """Tests for BatchPredictResponse schema."""

    def test_valid_batch_response(self):
        """Should accept valid batch data."""
        resp = BatchPredictResponse(
            results=[],
            total_files=0,
            total_execution_time_seconds=0.0,
        )
        assert resp.total_files == 0
        assert resp.results == []


class TestModelInfoResponse:
    """Tests for ModelInfoResponse schema."""

    def test_valid_model_info(self):
        """Should accept valid model info data."""
        resp = ModelInfoResponse(
            model_name="MultiModalModel",
            vision_backbone="convnext_base",
            text_backbone="ModernBERT-base",
            document_classes=16,
            clause_classes=41,
            device="cpu",
            total_parameters=100000,
            trainable_parameters=50000,
        )
        assert resp.model_name == "MultiModalModel"
        assert resp.total_parameters == 100000


class TestVersionResponse:
    """Tests for VersionResponse schema."""

    def test_valid_version(self):
        """Should accept valid version data."""
        resp = VersionResponse(
            api_version="1.0.0",
            python_version="3.11.0",
            pytorch_version="2.0.0",
            cuda_available=False,
        )
        assert resp.api_version == "1.0.0"
        assert resp.cuda_version is None

    def test_with_cuda_version(self):
        """Should accept CUDA version when available."""
        resp = VersionResponse(
            api_version="1.0.0",
            python_version="3.11.0",
            pytorch_version="2.0.0",
            cuda_available=True,
            cuda_version="12.1",
        )
        assert resp.cuda_version == "12.1"


class TestErrorResponse:
    """Tests for ErrorResponse schema."""

    def test_valid_error(self):
        """Should accept valid error data."""
        resp = ErrorResponse(
            error="Bad request",
            detail="Invalid file type",
            status_code=400,
        )
        assert resp.error == "Bad request"
        assert resp.status_code == 400

    def test_default_status_code(self):
        """Default status code should be 500."""
        resp = ErrorResponse(error="Internal error")
        assert resp.status_code == 500
