"""Settings page."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient


def render_settings(client: APIClient) -> None:
    """Render the settings page.

    Provides API configuration and application information.

    Args:
        client: APIClient instance for backend communication.
    """
    st.markdown("## ⚙️ Settings")

    # ---- API Configuration ----
    st.markdown("### API Configuration")

    current_url = st.session_state.get("api_base_url", "http://localhost:8000")

    new_url = st.text_input(
        "API Base URL",
        value=current_url,
        help="Base URL of the FastAPI backend server",
        key="settings_api_url",
    )

    if new_url != current_url:
        st.session_state["api_base_url"] = new_url
        st.info(
            f"API URL updated to: **{new_url}**. "
            f"Changes take effect on next request."
        )

    # Test connection
    if st.button("🔗 Test Connection", key="test_connection"):
        test_client = APIClient(base_url=new_url)
        if test_client.is_available():
            st.success("✅ Connection successful!")
            try:
                health = test_client.health_check()
                st.json(health)
            except Exception:
                pass
        else:
            st.error(f"❌ Cannot connect to {new_url}")

    st.markdown("---")

    # ---- About ----
    st.markdown("### About")

    st.markdown(
        """
        <div class="card">
            <div style="color: #f1f5f9; line-height: 1.8;">
                <strong style="font-size: 1.1rem;">
                    ⚖️ Multimodal Legal Risk Auditor
                </strong>
                <br><br>
                A dual-stage machine learning system for analyzing legal contracts
                and documents. Combines textual information from contracts with
                visual/structural features of document page scans to detect and
                flag high-risk contract clauses.
                <br><br>
                <span style="color: #94a3b8;">
                    <strong>Version:</strong> 1.0.0<br>
                    <strong>ML Models:</strong> ConvNeXt + ModernBERT + Cross-Attention Fusion<br>
                    <strong>Clause Classes:</strong> 41 legal risk categories<br>
                    <strong>Document Types:</strong> 16 visual layout categories
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ---- Session State Debug ----
    with st.expander("🔧 Debug: Session State", expanded=False):
        st.json(
            {
                k: str(v)[:200] if isinstance(v, (dict, list, str)) else str(v)
                for k, v in st.session_state.items()
            }
        )
