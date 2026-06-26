"""Home dashboard page."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient


def render_home(client: APIClient) -> None:
    """Render the home dashboard page.

    Displays system status, quick-start guide, and overview metrics.

    Args:
        client: APIClient instance for backend communication.
    """
    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>⚖️ Legal Risk Auditor</h1>
            <p>Multimodal AI-powered contract risk analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # System Status
    col1, col2, col3 = st.columns(3)

    try:
        health = client.health_check()
        api_online = True
        device = health.get("device", "unknown")
        model_loaded = health.get("model_loaded", False)
    except Exception:
        api_online = False
        device = "N/A"
        model_loaded = False

    with col1:
        status_dot = "online" if api_online else "offline"
        status_text = "Online" if api_online else "Offline"
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">API Status</div>
                <div class="card-value">
                    <span class="status-dot {status_dot}"></span>
                    {status_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Device</div>
                <div class="card-value">{device.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        model_status = "Ready" if model_loaded else "Not Loaded"
        model_icon = "🟢" if model_loaded else "🔴"
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Model</div>
                <div class="card-value">{model_icon} {model_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick Start
    st.markdown("### 🚀 Quick Start")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">How It Works</div>
                <div style="color: #e2e8f0; line-height: 1.8;">
                    <strong>1.</strong> Upload a legal document (PDF, DOCX, or image)<br>
                    <strong>2.</strong> AI analyzes text content + visual layout<br>
                    <strong>3.</strong> Receive risk score, detected clauses, and recommendations<br>
                    <strong>4.</strong> Download reports in JSON, PDF, or HTML
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Supported Formats</div>
                <div style="color: #e2e8f0; line-height: 1.8;">
                    📄 <strong>PDF</strong> — Multi-page contract scans<br>
                    📝 <strong>DOCX</strong> — Microsoft Word documents<br>
                    🖼️ <strong>PNG / JPG</strong> — Document page images<br>
                    🖼️ <strong>TIFF / BMP</strong> — High-resolution scans
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Version info
    if api_online:
        try:
            version_info = client.version()
            st.markdown("---")
            st.markdown("### 📊 Runtime Environment")

            v_cols = st.columns(4)
            with v_cols[0]:
                st.metric("API Version", version_info.get("api_version", "N/A"))
            with v_cols[1]:
                st.metric("Python", version_info.get("python_version", "N/A"))
            with v_cols[2]:
                st.metric("PyTorch", version_info.get("pytorch_version", "N/A"))
            with v_cols[3]:
                cuda = "Available" if version_info.get("cuda_available") else "CPU Only"
                st.metric("CUDA", cuda)

        except Exception:
            pass
