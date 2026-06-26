from __future__ import annotations

import os
import sys
import time
import tempfile
import shutil
from typing import List

import torch
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Response

from src.api.dependencies import (
    get_predictor,
    get_processor,
)
from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    BatchPredictResponse,
    ModelInfoResponse,
    VersionResponse,
    ErrorResponse,
)
from src.utils.logging_config import get_api_logger


logger = get_api_logger()

router = APIRouter()


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx",
    ".png", ".jpg", ".jpeg",
    ".tif", ".tiff", ".bmp",
}


def _validate_file_extension(filename: str) -> str:
    """Validate uploaded file has a supported extension.

    Args:
        filename: Name of the uploaded file.

    Returns:
        The file extension (lowercase, with dot).

    Raises:
        HTTPException: If file extension is not supported.
    """
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )
    return ext


def _save_upload_to_temp(file: UploadFile, suffix: str) -> str:
    """Save an uploaded file to a temporary path.

    Args:
        file: The uploaded file.
        suffix: File extension suffix.

    Returns:
        Path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def _run_prediction(
    path: str,
    predictor,
    processor,
) -> dict:
    """Run the full prediction pipeline on a file.

    Args:
        path: Path to the file on disk.
        predictor: Predictor instance.
        processor: DocumentProcessor instance.

    Returns:
        Prediction result dictionary.
    """
    document = processor.process(path)
    result = predictor.predict(document)
    # Include OCR text in the result for the UI
    result["ocr_text"] = document.text
    return result


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health(
    predictor=Depends(get_predictor),
):
    """Check API health, device status, and model readiness.

    Returns:
        Health status including device and model loaded state.
    """
    return HealthResponse(
        status="ok",
        device=str(predictor.device),
        model_loaded=predictor.model is not None,
    )


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------

@router.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
)
async def predict(

    file: UploadFile = File(...),

    predictor=Depends(get_predictor),

    processor=Depends(get_processor),

):
    """Analyze a single document for legal risk.

    Accepts PDF, DOCX, PNG, JPG, JPEG, TIFF, or BMP files.

    Args:
        file: The uploaded document file.
        predictor: Injected Predictor singleton.
        processor: Injected DocumentProcessor singleton.

    Returns:
        Risk analysis results including score, level, clauses.
    """
    ext = _validate_file_extension(file.filename)
    path = _save_upload_to_temp(file, suffix=ext)

    try:
        start = time.perf_counter()

        result = _run_prediction(path, predictor, processor)

        elapsed = time.perf_counter() - start
        result["execution_time_seconds"] = round(elapsed, 4)

        logger.info(
            "Prediction complete: file=%s risk_level=%s time=%.4fs",
            file.filename,
            result.get("risk_level"),
            elapsed,
        )

        return result

    finally:
        # Clean up temporary file
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------
# Batch Predict
# ---------------------------------------------------------

@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    tags=["Prediction"],
)
async def batch_predict(

    files: List[UploadFile] = File(...),

    predictor=Depends(get_predictor),

    processor=Depends(get_processor),

):
    """Analyze multiple documents for legal risk in a single request.

    Args:
        files: List of uploaded document files.
        predictor: Injected Predictor singleton.
        processor: Injected DocumentProcessor singleton.

    Returns:
        Batch results with individual predictions and total timing.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file is required.",
        )

    batch_start = time.perf_counter()
    results = []

    for file in files:
        ext = _validate_file_extension(file.filename)
        path = _save_upload_to_temp(file, suffix=ext)

        try:
            start = time.perf_counter()
            result = _run_prediction(path, predictor, processor)
            elapsed = time.perf_counter() - start
            result["execution_time_seconds"] = round(elapsed, 4)
            results.append(result)

            logger.info(
                "Batch item: file=%s risk_level=%s time=%.4fs",
                file.filename,
                result.get("risk_level"),
                elapsed,
            )

        except Exception as exc:
            logger.error(
                "Batch item failed: file=%s error=%s",
                file.filename,
                str(exc),
            )
            results.append({
                "error": str(exc),
                "filename": file.filename,
                "risk_score": 0.0,
                "risk_level": "ERROR",
                "document_type": "unknown",
                "confidence": 0.0,
                "recommendation": f"Processing failed: {exc}",
                "high_risk_clauses": [],
            })

        finally:
            if os.path.exists(path):
                os.unlink(path)

    total_elapsed = time.perf_counter() - batch_start

    logger.info(
        "Batch complete: files=%d total_time=%.4fs",
        len(files),
        total_elapsed,
    )

    return BatchPredictResponse(
        results=results,
        total_files=len(files),
        total_execution_time_seconds=round(total_elapsed, 4),
    )


# ---------------------------------------------------------
# Model Info
# ---------------------------------------------------------

@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
)
def model_info(
    predictor=Depends(get_predictor),
):
    """Get model architecture details and parameter counts.

    Returns:
        Model information including backbone names, parameter counts, device.
    """
    model = predictor.model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return ModelInfoResponse(
        model_name="MultiModalModel",
        vision_backbone="convnext_base.fb_in22k_ft_in1k",
        text_backbone="answerdotai/ModernBERT-base",
        document_classes=6,
        clause_classes=41,
        device=str(predictor.device),
        total_parameters=total_params,
        trainable_parameters=trainable_params,
    )


# ---------------------------------------------------------
# Version
# ---------------------------------------------------------

@router.get(
    "/version",
    response_model=VersionResponse,
    tags=["System"],
)
def version():
    """Get API version and runtime environment information.

    Returns:
        Version info including API version, Python, PyTorch, CUDA.
    """
    return VersionResponse(
        api_version="1.0.0",
        python_version=sys.version.split()[0],
        pytorch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=(
            torch.version.cuda
            if torch.cuda.is_available()
            else None
        ),
    )


# ---------------------------------------------------------
# Report Downloads
# ---------------------------------------------------------

@router.post(
    "/report/json",
    tags=["Reports"],
)
async def download_json_report(result: dict):
    """Generate and download JSON risk report.

    Args:
        result: Prediction result dictionary.

    Returns:
        JSON file response.
    """
    try:
        from src.reports.json_report import generate_json_report
        json_data = generate_json_report(result)
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    "attachment; filename=risk_report.json"
                )
            },
        )
    except Exception as exc:
        logger.error("Failed to generate JSON report: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/report/pdf",
    tags=["Reports"],
)
async def download_pdf_report(result: dict):
    """Generate and download PDF risk report.

    Args:
        result: Prediction result dictionary.

    Returns:
        PDF file response.
    """
    try:
        from src.reports.pdf_report import generate_pdf_report
        pdf_bytes = generate_pdf_report(result)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    "attachment; filename=risk_report.pdf"
                )
            },
        )
    except Exception as exc:
        logger.error("Failed to generate PDF report: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/report/html",
    tags=["Reports"],
)
async def download_html_report(result: dict):
    """Generate and download HTML risk report.

    Args:
        result: Prediction result dictionary.

    Returns:
        HTML file response.
    """
    try:
        from src.reports.html_report import generate_html_report
        html_data = generate_html_report(result)
        return Response(
            content=html_data,
            media_type="text/html",
            headers={
                "Content-Disposition": (
                    "attachment; filename=risk_report.html"
                )
            },
        )
    except Exception as exc:
        logger.error("Failed to generate HTML report: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))