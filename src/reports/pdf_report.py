"""PDF report generation from inference results using reportlab."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)


_RISK_COLORS = {
    "LOW": colors.HexColor("#22c55e"),
    "MEDIUM": colors.HexColor("#eab308"),
    "HIGH": colors.HexColor("#f97316"),
    "CRITICAL": colors.HexColor("#ef4444"),
}

_BG_DARK = colors.HexColor("#0f172a")
_BG_CARD = colors.HexColor("#1e293b")
_TEXT_PRIMARY = colors.HexColor("#f1f5f9")
_TEXT_SECONDARY = colors.HexColor("#94a3b8")
_BORDER = colors.HexColor("#334155")


def _build_styles() -> dict:
    """Build paragraph styles for the PDF report.

    Returns:
        Dictionary of named ParagraphStyle objects.
    """
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=_TEXT_PRIMARY,
            spaceAfter=4 * mm,
            alignment=1,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=10,
            textColor=_TEXT_SECONDARY,
            spaceAfter=8 * mm,
            alignment=1,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=14,
            textColor=_TEXT_PRIMARY,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=10,
            textColor=_TEXT_PRIMARY,
            spaceAfter=2 * mm,
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base["Normal"],
            fontSize=9,
            textColor=_TEXT_SECONDARY,
        ),
        "value": ParagraphStyle(
            "Value",
            parent=base["Normal"],
            fontSize=11,
            textColor=_TEXT_PRIMARY,
        ),
        "risk_score": ParagraphStyle(
            "RiskScore",
            parent=base["Title"],
            fontSize=36,
            alignment=1,
            spaceAfter=2 * mm,
        ),
        "risk_level": ParagraphStyle(
            "RiskLevel",
            parent=base["Normal"],
            fontSize=16,
            alignment=1,
            spaceAfter=4 * mm,
        ),
        "recommendation": ParagraphStyle(
            "Recommendation",
            parent=base["Normal"],
            fontSize=11,
            textColor=_TEXT_PRIMARY,
            leftIndent=10 * mm,
            spaceAfter=4 * mm,
        ),
        "ocr": ParagraphStyle(
            "OcrText",
            parent=base["Code"],
            fontSize=7,
            textColor=_TEXT_SECONDARY,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
        ),
    }


def generate_pdf_report(result: Dict[str, Any]) -> bytes:
    """Generate a professional PDF report from prediction results.

    Args:
        result: Prediction result dictionary from the Predictor.

    Returns:
        PDF document as bytes.
    """
    buffer = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    risk_level = result.get("risk_level", "LOW")
    risk_score = result.get("risk_score", 0)
    risk_color = _RISK_COLORS.get(risk_level, _TEXT_SECONDARY)
    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    elements = []

    # ---- Title ----
    elements.append(
        Paragraph("Legal Risk Audit Report", styles["title"])
    )
    elements.append(
        Paragraph(
            f"{result.get('filename', 'Unknown Document')} — {generated_at}",
            styles["subtitle"],
        )
    )
    elements.append(
        HRFlowable(
            width="100%", thickness=1,
            color=_BORDER, spaceAfter=6 * mm,
        )
    )

    # ---- Risk Score ----
    score_style = ParagraphStyle(
        "DynRiskScore",
        parent=styles["risk_score"],
        textColor=risk_color,
    )
    level_style = ParagraphStyle(
        "DynRiskLevel",
        parent=styles["risk_level"],
        textColor=risk_color,
    )
    elements.append(Paragraph(str(risk_score), score_style))
    elements.append(
        Paragraph(f"{risk_level} RISK", level_style)
    )

    # ---- Recommendation ----
    elements.append(
        Paragraph(
            f"<b>Recommendation:</b> {result.get('recommendation', 'N/A')}",
            styles["recommendation"],
        )
    )
    elements.append(Spacer(1, 4 * mm))

    # ---- Document Info Table ----
    elements.append(
        Paragraph("Document Information", styles["heading"])
    )

    info_data = [
        ["Document Type", str(result.get("document_type", "N/A"))],
        ["Confidence", f"{result.get('confidence', 0)}%"],
        ["File Type", str(result.get("file_type", "N/A"))],
        ["Page Count", str(result.get("page_count", "N/A"))],
        ["OCR Engine", str(result.get("ocr_engine", "N/A"))],
        [
            "Processing Time",
            f"{result.get('execution_time_seconds', 'N/A')}s",
        ],
    ]

    info_table = Table(info_data, colWidths=[60 * mm, 100 * mm])
    info_table.setStyle(
        TableStyle([
            ("TEXTCOLOR", (0, 0), (0, -1), _TEXT_SECONDARY),
            ("TEXTCOLOR", (1, 0), (1, -1), _TEXT_PRIMARY),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
            ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
            (
                "LINEBELOW", (0, 0), (-1, -1),
                0.5, _BORDER,
            ),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    elements.append(info_table)
    elements.append(Spacer(1, 4 * mm))

    # ---- Detected Clauses ----
    elements.append(
        Paragraph("Detected High-Risk Clauses", styles["heading"])
    )

    clauses = result.get("high_risk_clauses", [])
    if clauses:
        clause_data = [["Clause", "Probability"]]
        for clause in clauses:
            clause_data.append([
                clause.get("clause", "Unknown"),
                f"{round(clause.get('probability', 0) * 100, 1)}%",
            ])

        clause_table = Table(
            clause_data,
            colWidths=[110 * mm, 50 * mm],
        )
        clause_table.setStyle(
            TableStyle([
                ("TEXTCOLOR", (0, 0), (-1, 0), _TEXT_SECONDARY),
                ("TEXTCOLOR", (0, 1), (-1, -1), _TEXT_PRIMARY),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                (
                    "LINEBELOW", (0, 0), (-1, -1),
                    0.5, _BORDER,
                ),
                (
                    "LINEBELOW", (0, 0), (-1, 0),
                    1, _TEXT_SECONDARY,
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        elements.append(clause_table)
    else:
        elements.append(
            Paragraph(
                "No high-risk clauses detected above threshold.",
                styles["body"],
            )
        )

    # ---- OCR Text ----
    ocr_text = result.get("ocr_text", "")
    if ocr_text:
        elements.append(Spacer(1, 4 * mm))
        elements.append(
            Paragraph("Extracted Text (OCR)", styles["heading"])
        )
        # Truncate very long OCR text for PDF readability
        display_text = ocr_text[:3000]
        if len(ocr_text) > 3000:
            display_text += "\n\n[... truncated ...]"
        # Escape XML special characters for reportlab
        display_text = (
            display_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        elements.append(
            Paragraph(display_text, styles["ocr"])
        )

    # ---- Footer ----
    elements.append(Spacer(1, 8 * mm))
    elements.append(
        HRFlowable(
            width="100%", thickness=0.5,
            color=_BORDER, spaceAfter=3 * mm,
        )
    )
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["label"],
        alignment=1,
    )
    elements.append(
        Paragraph(
            "Multimodal Legal Risk Auditor v1.0.0 — Automated Report",
            footer_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()
