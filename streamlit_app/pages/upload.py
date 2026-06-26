"""Upload page for document analysis."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient
from components.file_preview import render_file_preview


_ACCEPTED_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
]

_ACCEPTED_EXTENSIONS = ["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff", "bmp"]


def render_upload(client: APIClient) -> None:
    """Render the document upload page.

    Provides file upload, preview, and triggers prediction via the API.
    Results are stored in session state for the Results page.

    Args:
        client: APIClient instance for backend communication.
    """
    st.markdown("## 📤 Upload Document")
    st.markdown(
        '<p style="color: #94a3b8;">Upload a legal document for risk analysis</p>',
        unsafe_allow_html=True,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=_ACCEPTED_EXTENSIONS,
        help="Supported: PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP",
        key="file_uploader",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # Preview
        col_preview, col_info = st.columns([2, 1])

        with col_preview:
            st.markdown("### Preview")
            render_file_preview(file_bytes, filename)

        with col_info:
            st.markdown("### File Details")
            size_kb = len(file_bytes) / 1024
            ext = filename.rsplit(".", 1)[-1].upper() if "." in filename else "?"
            st.markdown(
                f"""
                <div class="card">
                    <div style="color: #94a3b8; font-size: 0.85rem;">Filename</div>
                    <div style="color: #f1f5f9; font-weight: 600;
                                word-break: break-all;">{filename}</div>
                    <br>
                    <div style="color: #94a3b8; font-size: 0.85rem;">Type</div>
                    <div style="color: #f1f5f9; font-weight: 600;">{ext}</div>
                    <br>
                    <div style="color: #94a3b8; font-size: 0.85rem;">Size</div>
                    <div style="color: #f1f5f9; font-weight: 600;">{size_kb:.1f} KB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Analyze button
        if st.button(
            "🔍 Analyze Document",
            use_container_width=True,
            key="analyze_btn",
        ):
            # Check API availability
            if not client.is_available():
                st.error(
                    "⚠️ Cannot connect to the API server. "
                    "Please ensure it is running."
                )
                return

            with st.spinner("🔄 Processing document..."):
                progress = st.progress(0, text="Uploading...")

                try:
                    progress.progress(20, text="Sending to API...")

                    result = client.predict(
                        file_bytes=file_bytes,
                        filename=filename,
                    )

                    progress.progress(90, text="Processing results...")

                    # Store results in session state
                    st.session_state["prediction_result"] = result
                    st.session_state["current_page"] = "Results"

                    progress.progress(100, text="Complete!")

                    st.success(
                        f"✅ Analysis complete! "
                        f"Risk Level: **{result.get('risk_level', 'N/A')}** "
                        f"(Score: {result.get('risk_score', 0)})"
                    )

                    st.info("Navigate to **Results** in the sidebar to view details.")

                except RuntimeError as exc:
                    st.error(f"❌ Analysis failed: {exc}")
                except Exception as exc:
                    st.error(f"❌ Unexpected error: {exc}")
