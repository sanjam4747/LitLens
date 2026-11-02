"""
Plotly chart components and styling for LitLens.
Adapts BookMind chart layouts for academic publication trends,
category distributions, and hybrid score contribution breakdowns.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = ["#58a6ff", "#3fb950", "#d2a8ff", "#f78166", "#ffa657", "#79c0ff", "#ff7b72"]

CHART_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", color="#e6edf3", size=12),
    margin=dict(t=40, b=40, l=40, r=20),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickfont=dict(color="#8b949e")),
)


def plot_category_distribution(category_series: pd.Series, top_n: int = 12) -> go.Figure:
    """Plots top scientific categories by paper count."""
    counts = category_series.value_counts().head(top_n)
    labels = [c[:38] + "…" if len(c) > 38 else c for c in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.values[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(
            color=counts.values[::-1],
            colorscale=[[0, "#1f6feb"], [1, "#58a6ff"]],
            showscale=False,
        ),
        text=counts.values[::-1],
        textposition="outside",
        textfont=dict(color="#8b949e", size=11),
    ))
    layout = dict(**CHART_LAYOUT)
    layout["margin"] = dict(t=20, b=30, l=180, r=40)
    fig.update_layout(**layout, height=380, xaxis_title="Indexed Papers", yaxis_title="")
    return fig


def plot_publication_timeline(years_series: pd.Series, min_year: int = 1993, max_year: int = 2026) -> go.Figure:
    """Plots publications timeline from 1993 to present."""
    valid_years = years_series[(years_series >= min_year) & (years_series <= max_year)]
    counts = valid_years.value_counts().sort_index().reset_index()
    counts.columns = ["Year", "Count"]

    fig = go.Figure(go.Scatter(
        x=counts["Year"],
        y=counts["Count"],
        mode="lines+markers",
        line=dict(color="#58a6ff", width=2.5),
        marker=dict(size=5, color="#58a6ff"),
        fill="tozeroy",
        fillcolor="rgba(88, 166, 255, 0.12)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=320,
        xaxis_title="Publication Year",
        yaxis_title="Papers Published",
    )
    return fig


def plot_score_contributions(recommendations: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders stacked bar chart decomposing Final Hybrid Score into
    TF-IDF contribution and Topic contribution.
    """
    if not recommendations:
        return go.Figure()

    df = pd.DataFrame(recommendations)
    titles = [
        f"#{i+1} " + (t[:26] + "…" if len(t) > 26 else t)
        for i, t in enumerate(df["title"])
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="TF-IDF Contribution",
        x=titles,
        y=df["tfidf_contribution"],
        marker_color="#58a6ff",
    ))
    fig.add_trace(go.Bar(
        name="Topic Contribution",
        x=titles,
        y=df["topic_contribution"],
        marker_color="#d2a8ff",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        barmode="stack",
        height=340,
        xaxis_title="",
        yaxis_title="Score Contribution",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(size=11),
        ),
    )
    return fig


def plot_topic_overview(topic_summaries: Dict[int, Dict[str, Any]], dominant_counts: Dict[int, int]) -> go.Figure:
    """Plots paper count distribution across the 16 LDA topics."""
    topic_ids = list(topic_summaries.keys())
    labels = [topic_summaries[k]["label"][:32] for k in topic_ids]
    counts = [dominant_counts.get(k, 0) for k in topic_ids]

    fig = go.Figure(go.Bar(
        x=counts[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(
            color=counts[::-1],
            colorscale=[[0, "#30363d"], [0.5, "#8957e5"], [1, "#d2a8ff"]],
            showscale=False,
        ),
        text=counts[::-1],
        textposition="outside",
        textfont=dict(color="#8b949e", size=10),
    ))
    layout = dict(**CHART_LAYOUT)
    layout["margin"] = dict(t=20, b=30, l=220, r=40)
    fig.update_layout(**layout, height=420, xaxis_title="Primary Papers", yaxis_title="")
    return fig

