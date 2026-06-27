"""HTTP client for communicating with the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import requests


class APIClient:
    """Client for the Multimodal Legal Risk Auditor FastAPI backend.

    Args:
        base_url:
            FastAPI base URL.
            Uses API_URL environment variable when running in Docker.
        timeout:
            Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
    ):

        self.base_url = (
            base_url
            or os.getenv(
                "API_URL",
                "http://localhost:8000",
            )
        ).rstrip("/")

        self.timeout = timeout

    ############################################################
    # Health
    ############################################################

    def health_check(self) -> Dict[str, Any]:

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

            raise ConnectionError(
                f"Health check failed: {exc}"
            )

    ############################################################
    # Single Prediction
    ############################################################

    def predict(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> Dict[str, Any]:

        try:

            files = {
                "file": (
                    filename,
                    file_bytes,
                ),
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

                detail = exc.response.json().get(
                    "detail",
                    "",
                )

            except Exception:
                pass

            raise RuntimeError(
                f"Prediction failed "
                f"({exc.response.status_code}): "
                f"{detail}"
            )

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Prediction request failed: {exc}"
            )

    ############################################################
    # Batch Prediction
    ############################################################

    def batch_predict(
        self,
        file_list: List[tuple],
    ) -> Dict[str, Any]:

        try:

            files = [

                (
                    "files",
                    (
                        name,
                        data,
                    ),
                )

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

                detail = exc.response.json().get(
                    "detail",
                    "",
                )

            except Exception:
                pass

            raise RuntimeError(
                f"Batch prediction failed "
                f"({exc.response.status_code}): "
                f"{detail}"
            )

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Batch request failed: {exc}"
            )

    ############################################################
    # Model Info
    ############################################################

    def model_info(self) -> Dict[str, Any]:

        try:

            response = requests.get(
                f"{self.base_url}/model/info",
                timeout=30,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Model info request failed: {exc}"
            )

    ############################################################
    # Version
    ############################################################

    def version(self) -> Dict[str, Any]:

        try:

            response = requests.get(
                f"{self.base_url}/version",
                timeout=10,
            )

            response.raise_for_status()

            return response.json()

        except requests.RequestException as exc:

            raise RuntimeError(
                f"Version request failed: {exc}"
            )

    ############################################################
    # Availability
    ############################################################

    def is_available(self) -> bool:

        try:

            self.health_check()

            return True

        except Exception:

            return False