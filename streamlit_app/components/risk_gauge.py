"""Risk score gauge visualization using Plotly."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go


_RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}


def render_risk_gauge(
    score: float,
    risk_level: str,
    height: int = 280,
) -> go.Figure:
    """Create a Plotly gauge chart for risk score visualization.

    Args:
        score: Risk score from 0 to 100.
        risk_level: Risk level string (LOW, MEDIUM, HIGH, CRITICAL).
        height: Chart height in pixels.

    Returns:
        Plotly Figure object.
    """
    bar_color = _RISK_COLORS.get(risk_level, "#6b7280")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={
                "suffix": "",
                "font": {"size": 48, "color": bar_color},
            },
            title={
                "text": f"<b>{risk_level}</b> Risk",
                "font": {"size": 18, "color": bar_color},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#334155",
                    "dtick": 25,
                    "tickfont": {"color": "#94a3b8", "size": 11},
                },
                "bar": {"color": bar_color, "thickness": 0.7},
                "bgcolor": "#1e293b",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(34, 197, 94, 0.1)"},
                    {"range": [25, 50], "color": "rgba(234, 179, 8, 0.1)"},
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.1)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.1)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )

    return fig
