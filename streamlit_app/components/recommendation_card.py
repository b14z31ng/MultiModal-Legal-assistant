"""Recommendation card component."""

from __future__ import annotations

import streamlit as st


def render_recommendation_card(
    recommendation: str,
    risk_level: str,
) -> None:
    """Render a styled recommendation card with risk-level color coding.

    Args:
        recommendation: Recommendation text from RiskEngine.
        risk_level: Risk level (LOW, MEDIUM, HIGH, CRITICAL).
    """
    level_lower = risk_level.lower()

    colors = {
        "low": "#22c55e",
        "medium": "#eab308",
        "high": "#f97316",
        "critical": "#ef4444",
    }
    border_color = colors.get(level_lower, "#6366f1")

    icons = {
        "low": "✅",
        "medium": "⚠️",
        "high": "🔶",
        "critical": "🚨",
    }
    icon = icons.get(level_lower, "ℹ️")

    html = f"""
    <div style="
        background: #1e293b;
        border-left: 4px solid {border_color};
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    ">
        <div style="
            font-size: 0.8rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        ">{icon} Recommendation</div>
        <div style="
            font-size: 1.05rem;
            color: #f1f5f9;
            line-height: 1.5;
        ">{recommendation}</div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)
