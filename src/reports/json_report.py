"""JSON report generation from inference results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict


def generate_json_report(result: Dict[str, Any]) -> str:
    """Generate a structured JSON report from prediction results.

    Args:
        result: Prediction result dictionary from the Predictor.

    Returns:
        Formatted JSON string with full report data.
    """
    report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "Legal Risk Audit",
            "version": "1.0.0",
        },
        "document_info": {
            "filename": result.get("filename"),
            "file_type": result.get("file_type"),
            "source": result.get("source"),
            "page_count": result.get("page_count"),
            "ocr_engine": result.get("ocr_engine"),
        },
        "risk_assessment": {
            "risk_score": result.get("risk_score"),
            "risk_level": result.get("risk_level"),
            "recommendation": result.get("recommendation"),
        },
        "document_classification": {
            "document_type": result.get("document_type"),
            "confidence": result.get("confidence"),
        },
        "detected_clauses": result.get("high_risk_clauses", []),
        "extracted_text": result.get("ocr_text"),
        "execution": {
            "execution_time_seconds": result.get(
                "execution_time_seconds"
            ),
        },
    }

    return json.dumps(report, indent=2, ensure_ascii=False)
