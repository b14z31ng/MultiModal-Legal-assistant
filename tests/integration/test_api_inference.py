"""Integration tests for the API inference pipeline."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestPredictionPipeline:
    """Integration tests for the full prediction pipeline via API."""

    def test_upload_and_predict_image(
        self,
        test_client: TestClient,
        sample_image_bytes: bytes,
    ):
        """Full pipeline: upload image → get risk result."""
        response = test_client.post(
            "/predict",
            files={"file": ("contract.png", sample_image_bytes)},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify complete response structure
        assert "risk_score" in data
        assert "risk_level" in data
        assert "document_type" in data
        assert "confidence" in data
        assert "recommendation" in data
        assert "high_risk_clauses" in data
        assert "execution_time_seconds" in data

        # Verify types
        assert isinstance(data["risk_score"], (int, float))
        assert isinstance(data["risk_level"], str)
        assert isinstance(data["high_risk_clauses"], list)
        assert isinstance(data["execution_time_seconds"], float)

    def test_upload_and_predict_pdf(
        self,
        test_client: TestClient,
        sample_pdf_bytes: bytes,
    ):
        """Full pipeline: upload PDF → get risk result."""
        response = test_client.post(
            "/predict",
            files={"file": ("contract.pdf", sample_pdf_bytes)},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_batch_predict_returns_all_results(
        self,
        test_client: TestClient,
        sample_image_bytes: bytes,
    ):
        """Batch pipeline: upload 3 files → get 3 results."""
        files = [
            ("files", ("doc1.png", sample_image_bytes)),
            ("files", ("doc2.png", sample_image_bytes)),
            ("files", ("doc3.png", sample_image_bytes)),
        ]
        response = test_client.post("/batch", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        assert len(data["results"]) == 3

    def test_health_then_predict_workflow(
        self,
        test_client: TestClient,
        sample_image_bytes: bytes,
    ):
        """Workflow: check health → predict → verify."""
        # Step 1: Health check
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        # Step 2: Predict
        predict = test_client.post(
            "/predict",
            files={"file": ("test.png", sample_image_bytes)},
        )
        assert predict.status_code == 200

        # Step 3: Verify model info
        info = test_client.get("/model/info")
        assert info.status_code == 200
