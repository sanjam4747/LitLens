"""
Research Discovery page for LitLens Streamlit Dashboard.
Handles natural-language queries, interactive weight fusion, category & year filtering,
and explainable recommendations.
"""

import time
from typing import List, Dict, Any
import streamlit as st
import pandas as pd
from dashboard.components.cards import render_paper_card
from dashboard.components.charts import plot_score_contributions
from src.recommendation.hybrid_engine import HybridRecommender


SAMPLE_QUERIES = [
    "deep residual networks for image recognition and feature extraction",
    "prompt engineering in large language models for zero-shot reasoning",
    "reinforcement learning for robotic manipulation and continuous control",
    "graph convolutional networks for node classification and link prediction",
    "adversarial attacks and defense robustness in deep neural networks",
]


def render_discovery_page(hybrid: HybridRecommender, catalog: pd.DataFrame, top_k_setting: int):
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg, #1e3a5f 0%, #0d3b66 50%, #1d4ed8 100%);">
        <h1>🔍 Natural-Language Research Discovery</h1>
        <p>Search across 287k papers using free-form research queries. Combines exact keyword matching
        with latent scientific topic alignment.</p>
    </div>""", unsafe_allow_html=True)

    col_search, col_controls = st.columns([2, 1], gap="large")

    with col_search:
        st.markdown('<div class="section-header">Research Query</div>', unsafe_allow_html=True)
        sample_choice = st.selectbox(
            "Load an example research query:",
            ["Custom Query..."] + SAMPLE_QUERIES,
            index=0,
        )

        default_text = "" if sample_choice == "Custom Query..." else sample_choice
        query_text = st.text_input(
            "Enter research query, keywords, or abstract snippet:",
            value=default_text,
            placeholder="e.g. self-attention mechanisms in vision transformers",
        )

    with col_controls:
        st.markdown('<div class="section-header">⚙️ Model Fusion Weights</div>', unsafe_allow_html=True)
        w_tfidf = st.slider("TF-IDF Lexical Weight", 0.0, 1.0, 0.60, 0.05, help="Weights exact keyword matching")
        w_topic = st.slider("LDA Topic Weight", 0.0, 1.0, 0.40, 0.05, help="Weights thematic topic alignment")
        total_w = w_tfidf + w_topic
        if abs(total_w - 1.0) > 0.01 and total_w > 0:
            st.caption(f"ℹ️ Weights sum to {total_w:.2f} — auto-normalized during ranking.")

    # Filter row
    f1, f2 = st.columns(2)
    with f1:
        categories = ["All Categories"] + sorted(catalog["category"].dropna().unique().tolist())
        selected_category = st.selectbox("Filter by Category:", categories)
        cat_filter = None if selected_category == "All Categories" else selected_category

    with f2:
        min_yr = int(catalog["published_year"].min())
        max_yr = int(catalog["published_year"].max())
        year_range = st.slider("Publication Year Range:", min_yr, max_yr, (min_yr, max_yr))

    st.markdown("<br>", unsafe_allow_html=True)

    if not query_text.strip():
        st.markdown("""
        <div class="info-box" style="text-align:center;padding:30px;">
            💡 Enter a research query above or pick an example to generate hybrid recommendations.
        </div>""", unsafe_allow_html=True)
        return

    t0 = time.perf_counter()
    with st.spinner("Executing hybrid discovery across 287k papers…"):
        results = hybrid.recommend_for_query(
            query_text=query_text,
            top_k=top_k_setting,
            weight_tfidf=w_tfidf,
            weight_topic=w_topic,
            category=cat_filter,
            year_min=year_range[0],
            year_max=year_range[1],
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    if not results:
        st.warning("No matching research papers found. Try adjusting keywords or broadening filters.")
        return

    st.markdown(f'<div class="section-header">Top {len(results)} Research Recommendations</div>', unsafe_allow_html=True)
    st.caption(f"⚡ Live Measured Latency: **{latency_ms:.1f} ms** across 287,421 papers | Ranked by combined lexical and topic alignment")

    # Render paper cards
    for idx, r in enumerate(results, 1):
        render_paper_card(r, rank=idx)

    # Score breakdown visualization
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚖️ Score Contribution Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Decomposition of hybrid score into TF-IDF and Topic arms</div>', unsafe_allow_html=True)
    fig_contrib = plot_score_contributions(results)
    st.plotly_chart(fig_contrib, use_container_width=True)

