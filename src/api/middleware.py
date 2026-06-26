"""
FastAPI middleware for request logging and error handling.

Captures HTTP request details (method, path, status, duration)
and logs errors with stack traces.
"""

from __future__ import annotations

import time
import traceback

from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.utils.logging_config import get_request_logger, get_error_logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with method, path, status, and duration."""

    def __init__(self, app):
        super().__init__(app)
        self.request_logger = get_request_logger()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and log details.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/route handler.

        Returns:
            HTTP response.
        """
        start_time = time.perf_counter()
        client_ip = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start_time
            self.request_logger.error(
                "%s %s | client=%s | duration=%.4fs | status=500 (unhandled)",
                request.method,
                request.url.path,
                client_ip,
                duration,
            )
            raise

        duration = time.perf_counter() - start_time

        self.request_logger.info(
            "%s %s | client=%s | duration=%.4fs | status=%d",
            request.method,
            request.url.path,
            client_ip,
            duration,
            response.status_code,
        )

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns structured error responses."""

    def __init__(self, app):
        super().__init__(app)
        self.error_logger = get_error_logger()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with error boundary.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/route handler.

        Returns:
            HTTP response or error response.
        """
        try:
            return await call_next(request)
        except Exception as exc:
            self.error_logger.error(
                "Unhandled exception on %s %s: %s\n%s",
                request.method,
                request.url.path,
                str(exc),
                traceback.format_exc(),
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc),
                    "status_code": 500,
                },
            )
