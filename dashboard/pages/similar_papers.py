"""
Similar Papers ("More Like This") page for LitLens Streamlit Dashboard.
Retrieves related research given an indexed seed paper, strictly excluding
all revisions and versions sharing the seed paper's base_id.
"""

import time
import streamlit as st
import pandas as pd
from dashboard.components.cards import render_paper_card
from dashboard.components.charts import plot_score_contributions
from src.recommendation.hybrid_engine import HybridRecommender


def render_similar_papers_page(hybrid: HybridRecommender, catalog: pd.DataFrame, top_k_setting: int):
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg, #1b4d3e 0%, #206a5d 50%, #2d8a74 100%);">
        <h1>📖 Similar Paper Discovery ("More Like This")</h1>
        <p>Find research papers closely related to a specific reference work. The engine automatically
        excludes duplicate versions and earlier revisions of the selected paper.</p>
    </div>""", unsafe_allow_html=True)

    col_select, col_meta = st.columns([2, 1], gap="large")

    with col_select:
        st.markdown('<div class="section-header">Select a Seed Paper</div>', unsafe_allow_html=True)
        search_kw = st.text_input("🔎 Search seed paper by title or arXiv ID:", placeholder="e.g. Attention, ResNet, Backtracking, BERT...")

        if search_kw.strip():
            mask = (
                catalog["title"].str.contains(search_kw, case=False, na=False)
                | catalog["id"].str.contains(search_kw, case=False, na=False)
            )
            candidate_papers = catalog[mask].head(150).reset_index(drop=True)
        else:
            # Default to representative sample across major categories
            candidate_papers = catalog.sample(150, random_state=42).reset_index(drop=True)

        if candidate_papers.empty:
            st.warning("No papers match your search term. Try another title keyword.")
            return

        paper_options = [
            f"{r['title'][:65]}... ({r['id']})" if len(r['title']) > 65 else f"{r['title']} ({r['id']})"
            for _, r in candidate_papers.iterrows()
        ]
        sel_idx = st.selectbox(
            "Choose a paper from candidates:",
            range(len(paper_options)),
            format_func=lambda i: paper_options[i],
        )
        selected_paper = candidate_papers.iloc[sel_idx]
        seed_id = selected_paper["id"]

    with col_meta:
        st.markdown('<div class="section-header">Selected Paper Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px 20px;">
            <span class="category-pill">📄 {selected_paper['id']}</span>
            <span class="category-pill">🏷️ {selected_paper['category']}</span>
            <div style="font-size:1.02rem;font-weight:600;color:#f0f6fc;margin:8px 0 4px;">{selected_paper['title']}</div>
            <div style="font-size:0.80rem;color:#8b949e;">✍️ {selected_paper['authors_display']} · 📅 {selected_paper['published_year']}</div>
        </div>""", unsafe_allow_html=True)

    # Weights and filters
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        w_tfidf = st.slider("Lexical Match Weight", 0.0, 1.0, 0.60, 0.05, key="sim_tfidf")
    with c2:
        w_topic = st.slider("Topic Alignment Weight", 0.0, 1.0, 0.40, 0.05, key="sim_topic")
    with c3:
        filter_same_cat = st.checkbox("Constrain to same category only", value=False)
        cat_filter = selected_paper["category"] if filter_same_cat else None

    t0 = time.perf_counter()
    with st.spinner(f"Computing paper-to-paper similarities for {seed_id}…"):
        similar_recs = hybrid.recommend_for_paper(
            paper_id=seed_id,
            top_k=top_k_setting,
            weight_tfidf=w_tfidf,
            weight_topic=w_topic,
            category=cat_filter,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    if not similar_recs:
        st.info("No similar papers found matching the specified filter criteria.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">Similar Research to <i>"{selected_paper["title"][:45]}..."</i></div>', unsafe_allow_html=True)
    st.caption(f"⚡ Live Measured Latency: **{latency_ms:.1f} ms** across 287,421 papers | Source and version revisions strictly excluded")

    for idx, r in enumerate(similar_recs, 1):
        render_paper_card(r, rank=idx)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚖️ Score Composition</div>', unsafe_allow_html=True)
    fig_score = plot_score_contributions(similar_recs)
    st.plotly_chart(fig_score, use_container_width=True)

