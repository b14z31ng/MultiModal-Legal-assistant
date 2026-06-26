"""Detected clause table component."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


_RISK_COLORS = {
    "LOW": "#22c55e",
    "MEDIUM": "#eab308",
    "HIGH": "#f97316",
    "CRITICAL": "#ef4444",
}


def render_clause_table(
    clauses: List[Dict[str, Any]],
) -> None:
    """Render a styled table of detected high-risk clauses.

    Displays each clause with its name and a color-coded probability bar.
    Clauses are sorted by probability (highest first).

    Args:
        clauses: List of clause dictionaries with 'clause' and 'probability' keys.
    """
    if not clauses:
        st.info("✅ No high-risk clauses detected above threshold.")
        return

    sorted_clauses = sorted(
        clauses,
        key=lambda c: c.get("probability", 0),
        reverse=True,
    )

    # Build HTML table
    rows = ""
    for clause in sorted_clauses:
        name = clause.get("clause", "Unknown")
        prob = clause.get("probability", 0)
        pct = round(prob * 100, 1)

        # Determine bar color based on probability
        if pct >= 75:
            bar_color = _RISK_COLORS["CRITICAL"]
        elif pct >= 50:
            bar_color = _RISK_COLORS["HIGH"]
        elif pct >= 25:
            bar_color = _RISK_COLORS["MEDIUM"]
        else:
            bar_color = _RISK_COLORS["LOW"]

        rows += f"""
        <tr>
            <td style="padding: 0.75rem 1rem; border-bottom: 1px solid #334155;
                       color: #f1f5f9; font-size: 0.9rem;">{name}</td>
            <td style="padding: 0.75rem 1rem; border-bottom: 1px solid #334155;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="flex: 1; background: #334155; border-radius: 4px;
                                height: 8px; max-width: 180px;">
                        <div style="width: {pct}%; background: {bar_color};
                                    height: 100%; border-radius: 4px;
                                    transition: width 0.5s ease;"></div>
                    </div>
                    <span style="color: #94a3b8; font-size: 0.85rem;
                                 min-width: 45px;">{pct}%</span>
                </div>
            </td>
        </tr>"""

    html = f"""
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr>
                <th style="padding: 0.75rem 1rem; text-align: left;
                           color: #94a3b8; font-size: 0.8rem;
                           text-transform: uppercase; letter-spacing: 0.05em;
                           border-bottom: 2px solid #334155;">Clause</th>
                <th style="padding: 0.75rem 1rem; text-align: left;
                           color: #94a3b8; font-size: 0.8rem;
                           text-transform: uppercase; letter-spacing: 0.05em;
                           border-bottom: 2px solid #334155;">Probability</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    """

    st.markdown(html, unsafe_allow_html=True)
