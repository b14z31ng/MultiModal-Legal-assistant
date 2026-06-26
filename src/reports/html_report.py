"""HTML report generation from inference results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


_RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}


def generate_html_report(result: Dict[str, Any]) -> str:
    """Generate a standalone HTML report from prediction results.

    Produces a responsive, printable HTML document with embedded CSS,
    risk gauge visualization, and detected clause table.

    Args:
        result: Prediction result dictionary from the Predictor.

    Returns:
        Complete HTML document string.
    """
    risk_level = result.get("risk_level", "LOW")
    risk_score = result.get("risk_score", 0)
    risk_color = _RISK_COLORS.get(risk_level, "#6b7280")
    clauses = result.get("high_risk_clauses", [])
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    clause_rows = ""
    for clause in clauses:
        prob = clause.get("probability", 0)
        prob_pct = round(prob * 100, 1)
        clause_rows += f"""
        <tr>
            <td>{clause.get("clause", "Unknown")}</td>
            <td>
                <div class="prob-bar-bg">
                    <div class="prob-bar" style="width: {prob_pct}%;"></div>
                </div>
                <span class="prob-text">{prob_pct}%</span>
            </td>
        </tr>"""

    if not clause_rows:
        clause_rows = """
        <tr>
            <td colspan="2" style="text-align: center; color: #9ca3af;">
                No high-risk clauses detected above threshold.
            </td>
        </tr>"""

    ocr_text = result.get("ocr_text", "")
    ocr_section = ""
    if ocr_text:
        escaped_text = (
            ocr_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        ocr_section = f"""
        <div class="section">
            <h2>Extracted Text (OCR)</h2>
            <pre class="ocr-text">{escaped_text}</pre>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Legal Risk Audit Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 2rem 0;
            border-bottom: 1px solid #334155;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            font-size: 1.8rem;
            color: #f8fafc;
            margin-bottom: 0.5rem;
        }}
        .header .subtitle {{
            color: #94a3b8;
            font-size: 0.9rem;
        }}
        .section {{
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .section h2 {{
            font-size: 1.1rem;
            color: #f1f5f9;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #334155;
        }}
        .risk-gauge {{
            text-align: center;
            padding: 2rem;
        }}
        .risk-score {{
            font-size: 3.5rem;
            font-weight: 700;
            color: {risk_color};
        }}
        .risk-level {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {risk_color};
            text-transform: uppercase;
            margin-top: 0.5rem;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }}
        .info-item {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
        }}
        .info-label {{
            color: #94a3b8;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .info-value {{
            color: #f1f5f9;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.25rem;
        }}
        .recommendation {{
            background: #0f172a;
            border-left: 4px solid {risk_color};
            padding: 1rem 1.5rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{
            color: #94a3b8;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .prob-bar-bg {{
            background: #334155;
            border-radius: 4px;
            height: 8px;
            width: 100%;
            display: inline-block;
            max-width: 150px;
            vertical-align: middle;
        }}
        .prob-bar {{
            background: {risk_color};
            height: 100%;
            border-radius: 4px;
        }}
        .prob-text {{
            margin-left: 0.5rem;
            color: #94a3b8;
            font-size: 0.85rem;
        }}
        .ocr-text {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.8rem;
            max-height: 400px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #cbd5e1;
        }}
        .footer {{
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #334155;
        }}
        @media print {{
            body {{ background: #fff; color: #1e293b; padding: 1rem; }}
            .section {{ background: #f8fafc; border: 1px solid #e2e8f0; }}
            .info-item {{ background: #f1f5f9; }}
            .ocr-text {{ background: #f8fafc; color: #334155; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Legal Risk Audit Report</h1>
        <div class="subtitle">
            {result.get("filename", "Unknown Document")} &mdash;
            Generated {generated_at}
        </div>
    </div>

    <div class="section">
        <div class="risk-gauge">
            <div class="risk-score">{risk_score}</div>
            <div class="risk-level">{risk_level} Risk</div>
        </div>
        <div class="recommendation">
            <strong>Recommendation:</strong> {result.get("recommendation", "N/A")}
        </div>
    </div>

    <div class="section">
        <h2>Document Information</h2>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Document Type</div>
                <div class="info-value">{result.get("document_type", "N/A")}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Confidence</div>
                <div class="info-value">{result.get("confidence", 0)}%</div>
            </div>
            <div class="info-item">
                <div class="info-label">File Type</div>
                <div class="info-value">{result.get("file_type", "N/A")}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Page Count</div>
                <div class="info-value">{result.get("page_count", "N/A")}</div>
            </div>
            <div class="info-item">
                <div class="info-label">OCR Engine</div>
                <div class="info-value">{result.get("ocr_engine", "N/A")}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Processing Time</div>
                <div class="info-value">{result.get("execution_time_seconds", "N/A")}s</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Detected High-Risk Clauses</h2>
        <table>
            <thead>
                <tr>
                    <th>Clause</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>
                {clause_rows}
            </tbody>
        </table>
    </div>

    {ocr_section}

    <div class="footer">
        Multimodal Legal Risk Auditor v1.0.0 &mdash; Automated Report
    </div>
</body>
</html>"""

    return html
