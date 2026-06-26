"""
Multimodal Legal Risk Auditor — Streamlit Dashboard

Main entry point for the Streamlit application.
Run with: streamlit run streamlit_app/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from api_client import APIClient
from pages.home import render_home
from pages.upload import render_upload
from pages.results import render_results
from pages.model_info import render_model_info
from pages.settings import render_settings

# ---- Page Config ----
st.set_page_config(
    page_title="Legal Risk Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Load Custom CSS ----
_CSS_PATH = Path(__file__).parent / "styles" / "theme.css"
if _CSS_PATH.exists():
    st.markdown(
        f"<style>{_CSS_PATH.read_text()}</style>",
        unsafe_allow_html=True,
    )


# ---- Initialize API Client ----
def _get_client() -> APIClient:
    """Get or create the API client singleton.

    Returns:
        APIClient instance configured with the current base URL.
    """
    base_url = st.session_state.get(
        "api_base_url",
        "http://localhost:8000",
    )
    return APIClient(base_url=base_url)


# ---- Sidebar Navigation ----
_PAGES = {
    "🏠 Home": "Home",
    "📤 Upload": "Upload",
    "📊 Results": "Results",
    "🧠 Model Info": "Model Info",
    "⚙️ Settings": "Settings",
}

with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">⚖️</div>
            <div style="
                font-size: 1rem;
                font-weight: 700;
                background: linear-gradient(135deg, #6366f1, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-top: 0.25rem;
            ">Legal Risk Auditor</div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">
                v1.0.0
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Navigation radio
    selected = st.radio(
        "Navigation",
        list(_PAGES.keys()),
        label_visibility="collapsed",
        key="nav_radio",
    )

    current_page = _PAGES[selected]

    st.markdown("---")

    # Status indicator
    client = _get_client()
    try:
        health = client.health_check()
        st.markdown(
            '<div style="text-align: center;">'
            '<span class="status-dot online"></span>'
            '<span style="color: #94a3b8; font-size: 0.85rem;">API Connected</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            '<div style="text-align: center;">'
            '<span class="status-dot offline"></span>'
            '<span style="color: #94a3b8; font-size: 0.85rem;">API Offline</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---- Render Current Page ----
if current_page == "Home":
    render_home(client)
elif current_page == "Upload":
    render_upload(client)
elif current_page == "Results":
    render_results(client)
elif current_page == "Model Info":
    render_model_info(client)
elif current_page == "Settings":
    render_settings(client)
