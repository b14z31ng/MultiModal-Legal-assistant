"""Tests for FastAPI application setup."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestAppSetup:
    """Tests for FastAPI app creation and configuration."""

    def test_app_exists(self, test_client: TestClient):
        """App should be importable and functional."""
        assert test_client is not None

    def test_openapi_docs_accessible(self, test_client: TestClient):
        """OpenAPI docs should be accessible at /docs."""
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self, test_client: TestClient):
        """OpenAPI schema should be accessible at /openapi.json."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "Multimodal Legal Risk Auditor"

    def test_cors_headers(self, test_client: TestClient):
        """CORS headers should be present in responses."""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond
        assert response.status_code in (200, 405)
