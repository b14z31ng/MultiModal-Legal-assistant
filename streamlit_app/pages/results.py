"""Results page for displaying prediction outputs."""

from __future__ import annotations

import json

import streamlit as st

from api_client import APIClient
from components.risk_gauge import render_risk_gauge
from components.clause_table import render_clause_table
from components.recommendation_card import render_recommendation_card
from src.reports.json_report import generate_json_report
from src.reports.pdf_report import generate_pdf_report
from src.reports.html_report import generate_html_report


def render_results(client: APIClient) -> None:
    """Render the results page with full prediction output display.

    Shows risk gauge, recommendation, detected clauses, OCR text,
    inference metadata, and download buttons for reports.

    Args:
        client: APIClient instance (unused here, reserved for future use).
    """
    st.markdown("## 📊 Analysis Results")

    result = st.session_state.get("prediction_result")

    if result is None:
        st.info(
            "📤 No results yet. Upload a document on the **Upload** page first."
        )
        return

    risk_score = result.get("risk_score", 0)
    risk_level = result.get("risk_level", "LOW")
    filename = result.get("filename", "Unknown")

    st.markdown(
        f'<p style="color: #94a3b8;">Results for: <strong style="color: #f1f5f9;">'
        f'{filename}</strong></p>',
        unsafe_allow_html=True,
    )

    # ---- Row 1: Risk Gauge + Key Metrics ----
    col_gauge, col_metrics = st.columns([1, 1])

    with col_gauge:
        fig = render_risk_gauge(risk_score, risk_level)
        st.plotly_chart(fig, use_container_width=True, key="risk_gauge")

    with col_metrics:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Risk Score", f"{risk_score}")
            st.metric(
                "Document Type",
                result.get("document_type", "N/A"),
            )
        with m2:
            st.metric("Confidence", f"{result.get('confidence', 0)}%")
            st.metric(
                "Processing Time",
                f"{result.get('execution_time_seconds', 'N/A')}s",
            )

    # ---- Recommendation Card ----
    render_recommendation_card(
        recommendation=result.get("recommendation", "N/A"),
        risk_level=risk_level,
    )

    st.markdown("---")

    # ---- Tabs: Clauses | OCR Text | Metadata | Downloads ----
    tab_clauses, tab_ocr, tab_meta, tab_download = st.tabs(
        ["🔍 Detected Clauses", "📄 OCR Text", "ℹ️ Metadata", "📥 Downloads"]
    )

    with tab_clauses:
        clauses = result.get("high_risk_clauses", [])
        st.markdown(
            f"**{len(clauses)}** clause(s) detected above threshold"
        )
        render_clause_table(clauses)

    with tab_ocr:
        ocr_text = result.get("ocr_text", "")
        if ocr_text:
            st.markdown(
                f"""
                <div class="ocr-viewer">{
                    ocr_text
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                }</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("No OCR text available.")

    with tab_meta:
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.metric("File Type", result.get("file_type", "N/A"))
            st.metric("Filename", result.get("filename", "N/A"))
        with meta_cols[1]:
            st.metric("Page Count", result.get("page_count", "N/A"))
            st.metric("OCR Engine", result.get("ocr_engine", "N/A"))
        with meta_cols[2]:
            st.metric("Source", result.get("source", "N/A"))
            st.metric(
                "Execution Time",
                f"{result.get('execution_time_seconds', 'N/A')}s",
            )

    with tab_download:
        st.markdown("### Download Reports")

        dl_cols = st.columns(3)

        with dl_cols[0]:
            json_data = generate_json_report(result)
            st.download_button(
                label="📄 JSON Report",
                data=json_data,
                file_name=f"risk_report_{filename}.json",
                mime="application/json",
                use_container_width=True,
                key="dl_json",
            )

        with dl_cols[1]:
            try:
                pdf_data = generate_pdf_report(result)
                st.download_button(
                    label="📕 PDF Report",
                    data=pdf_data,
                    file_name=f"risk_report_{filename}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_pdf",
                )
            except Exception as exc:
                st.warning(f"PDF generation unavailable: {exc}")

        with dl_cols[2]:
            html_data = generate_html_report(result)
            st.download_button(
                label="🌐 HTML Report",
                data=html_data,
                file_name=f"risk_report_{filename}.html",
                mime="text/html",
                use_container_width=True,
                key="dl_html",
            )
