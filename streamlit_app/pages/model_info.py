"""Model information page."""

from __future__ import annotations

import streamlit as st

from api_client import APIClient


def render_model_info(client: APIClient) -> None:
    """Render the model information page.

    Displays model architecture details, parameter counts,
    backbone information, and device status.

    Args:
        client: APIClient instance for backend communication.
    """
    st.markdown("## 🧠 Model Information")

    if not client.is_available():
        st.error(
            "⚠️ Cannot connect to API server. "
            "Start the API to view model information."
        )
        return

    try:
        info = client.model_info()
    except Exception as exc:
        st.error(f"Failed to load model info: {exc}")
        return

    # ---- Architecture Overview ----
    st.markdown("### Architecture Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Model Name</div>
                <div class="card-value" style="font-size: 1.3rem;">
                    {info.get("model_name", "N/A")}
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
                <div class="card-value" style="font-size: 1.3rem;">
                    {info.get("device", "N/A").upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Backbones ----
    st.markdown("### Encoders")

    enc_col1, enc_col2 = st.columns(2)

    with enc_col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">🖼️ Vision Backbone</div>
                <div style="color: #f1f5f9; font-size: 1rem; font-weight: 600;
                            margin-top: 0.5rem;">
                    {info.get("vision_backbone", "N/A")}
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
                    ConvNeXt architecture for document layout feature extraction
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with enc_col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">📝 Text Backbone</div>
                <div style="color: #f1f5f9; font-size: 1rem; font-weight: 600;
                            margin-top: 0.5rem;">
                    {info.get("text_backbone", "N/A")}
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
                    ModernBERT encoder for contract text embeddings
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Parameters ----
    st.markdown("### Parameters")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    total = info.get("total_parameters", 0)
    trainable = info.get("trainable_parameters", 0)
    frozen = total - trainable

    with p_col1:
        st.metric(
            "Total Parameters",
            f"{total:,}",
        )
    with p_col2:
        st.metric(
            "Trainable",
            f"{trainable:,}",
        )
    with p_col3:
        st.metric(
            "Frozen",
            f"{frozen:,}",
        )
    with p_col4:
        st.metric(
            "Total (M)",
            f"{total / 1_000_000:.1f}M",
        )

    # ---- Output Heads ----
    st.markdown("### Output Heads")

    head_col1, head_col2 = st.columns(2)

    with head_col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Document Classes</div>
                <div class="card-value">{info.get("document_classes", "N/A")}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
                    Visual document type classification
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with head_col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="card-title">Clause Classes</div>
                <div class="card-value">{info.get("clause_classes", "N/A")}</div>
                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.25rem;">
                    Multi-label legal risk clause detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Architecture Diagram ----
    with st.expander("📐 Architecture Diagram", expanded=False):
        st.code(
            """
┌─────────────────┐     ┌──────────────────┐
│  Document Image  │     │  Contract Text   │
│   (384 × 384)   │     │  (512 tokens)    │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  ConvNeXt Base  │     │  ModernBERT Base │
│  Vision Encoder │     │   Text Encoder   │
│   (1024-dim)    │     │    (768-dim)     │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────┐
         │  Feature Fusion  │
         │  (1024-dim out)  │
         └────────┬─────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌────────┐
│Doc Head│  │Clause    │  │Risk    │
│(16 cls)│  │Head (41) │  │Head (1)│
└────────┘  └──────────┘  └────────┘
            """,
            language=None,
        )
