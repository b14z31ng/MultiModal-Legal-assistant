from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.middleware import (
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
)
from src.utils.logging_config import get_api_logger


logger = get_api_logger()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application startup and shutdown lifecycle handler.

    Args:
        application: The FastAPI application instance.
    """
    logger.info(
        "Starting Multimodal Legal Risk Auditor API (Python %s)",
        sys.version.split()[0],
    )
    yield
    logger.info("Shutting down API")


app = FastAPI(

    title="Multimodal Legal Risk Auditor",

    version="1.0.0",

    description=(
        "Production API for multimodal legal document risk analysis. "
        "Upload PDF, DOCX, or image files to receive risk assessments."
    ),

    lifespan=lifespan,

)


# ---------------------------------------------------------
# CORS — Allow Streamlit (port 8501) and local development
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Custom Middleware
# ---------------------------------------------------------
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------
@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError,
) -> JSONResponse:
    """Handle ValueError with a 400 response.

    Args:
        request: The incoming request.
        exc: The ValueError exception.

    Returns:
        JSON response with error details.
    """
    logger.warning("ValueError on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=400,
        content={
            "error": "Bad request",
            "detail": str(exc),
            "status_code": 400,
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(
    request: Request,
    exc: FileNotFoundError,
) -> JSONResponse:
    """Handle FileNotFoundError with a 404 response.

    Args:
        request: The incoming request.
        exc: The FileNotFoundError exception.

    Returns:
        JSON response with error details.
    """
    logger.warning("FileNotFoundError on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "detail": str(exc),
            "status_code": 404,
        },
    )


app.include_router(

    router,

)