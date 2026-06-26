"""HTTP client for communicating with the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


class APIClient:
    """Client for the Multimodal Legal Risk Auditor FastAPI backend.

    Provides methods for all API endpoints with error handling
    and configurable base URL.

    Args:
        base_url: Base URL of the FastAPI server.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> Dict[str, Any]:
        """Check API health status.

        Returns:
            Health response with status, device, and model_loaded.

        Raises:
            ConnectionError: If API is unreachable.
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to API at {self.base_url}"
            )
        except requests.RequestException as exc:
            raise ConnectionError(f"Health check failed: {exc}")

    def predict(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """Send a single file for prediction.

        Args:
            file_bytes: Raw file content bytes.
            filename: Name of the file being uploaded.

        Returns:
            Prediction result dictionary.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            files = {
                "file": (filename, file_bytes),
            }
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                pass
            raise RuntimeError(
                f"Prediction failed ({exc.response.status_code}): {detail}"
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Prediction request failed: {exc}")

    def batch_predict(
        self,
        file_list: List[tuple],
    ) -> Dict[str, Any]:
        """Send multiple files for batch prediction.

        Args:
            file_list: List of (filename, file_bytes) tuples.

        Returns:
            Batch prediction response.

        Raises:
            RuntimeError: If batch prediction fails.
        """
        try:
            files = [
                ("files", (name, data))
                for name, data in file_list
            ]
            response = requests.post(
                f"{self.base_url}/batch",
                files=files,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json().get("detail", "")
            except Exception:
                pass
            raise RuntimeError(
                f"Batch prediction failed ({exc.response.status_code}): {detail}"
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Batch request failed: {exc}")

    def model_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Model info including architecture and parameter counts.
        """
        try:
            response = requests.get(
                f"{self.base_url}/model/info",
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Model info request failed: {exc}")

    def version(self) -> Dict[str, Any]:
        """Get API version information.

        Returns:
            Version info including API, Python, PyTorch versions.
        """
        try:
            response = requests.get(
                f"{self.base_url}/version",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Version request failed: {exc}")

    def is_available(self) -> bool:
        """Check if the API is reachable.

        Returns:
            True if API responds to health check, False otherwise.
        """
        try:
            self.health_check()
            return True
        except (ConnectionError, Exception):
            return False
