"""
Page 2: Research Discovery View for LitLens Streamlit Dashboard.
Integrates:
- Tab 1: Natural-Language Research Query Search
- Tab 2: Similar Papers ("More Like This") with automatic base_id version exclusion
"""

import time
from typing import List, Dict, Any, Optional
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
    <div class="hero-banner" style="background:linear-gradient(135deg, #0e3056 0%, #0d3b66 45%, #1d4ed8 100%);">
        <h1>🔍 Research Discovery & Similar Papers</h1>
        <p>Explore 287k arXiv papers using free-form research queries or discover works similar to an indexed
        seed paper with automatic duplicate-version exclusion.</p>
    </div>""", unsafe_allow_html=True)

    tab_query, tab_similar = st.tabs(["🔎 Natural-Language Query Search", "📖 Similar Papers ('More Like This')"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: NATURAL LANGUAGE QUERY SEARCH
    # ══════════════════════════════════════════════════════════════════════════
    with tab_query:
        col_q_left, col_q_right = st.columns([1.6, 1.0], gap="medium")

        with col_q_left:
            st.markdown('<div class="section-header">Research Query</div>', unsafe_allow_html=True)
            preset_choice = st.selectbox(
                "Preset research queries:",
                ["(Select a preset or type below)"] + SAMPLE_QUERIES,
                index=0,
                key="preset_query_select",
            )
            default_query = "" if preset_choice.startswith("(") else preset_choice
            query_text = st.text_input(
                "Enter research topic, keywords, or abstract phrasing:",
                value=default_query,
                placeholder="e.g. self-attention mechanisms in vision transformers",
                key="query_input_text",
            )

        with col_q_right:
            st.markdown('<div class="section-header">⚙️ Model Fusion Weights</div>', unsafe_allow_html=True)
            w_tfidf = st.slider("TF-IDF Lexical Weight", 0.0, 1.0, 0.60, 0.05, key="w_tfidf_slider", help="Weight given to exact keyword & phrase matching")
            w_topic = st.slider("LDA Topic Weight", 0.0, 1.0, 0.40, 0.05, key="w_topic_slider", help="Weight given to broader thematic alignment")
            sum_w = w_tfidf + w_topic
            if abs(sum_w - 1.0) > 0.01 and sum_w > 0:
                st.caption(f"ℹ️ Weights sum to {sum_w:.2f} — auto-normalized during scoring.")

        # Filters Row
        fc1, fc2 = st.columns([1, 1], gap="medium")
        with fc1:
            cats = ["All Categories"] + sorted(catalog["category"].dropna().unique().tolist())
            sel_cat = st.selectbox("Category Filter:", cats, key="disc_cat_filter")
            cat_filter = None if sel_cat == "All Categories" else sel_cat
        with fc2:
            min_y = int(catalog["published_year"].min())
            max_y = int(catalog["published_year"].max())
            year_range = st.slider("Publication Year Range:", min_y, max_y, (min_y, max_y), key="disc_year_filter")

        if not query_text.strip():
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:24px;margin-top:14px;">
                💡 Enter a research query above or select a preset to generate hybrid recommendations.
            </div>""", unsafe_allow_html=True)
        else:
            t0 = time.perf_counter()
            with st.spinner("Executing hybrid retrieval across 287k papers…"):
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
                st.warning("No matching research papers found. Try broadening your keywords or filters.")
            else:
                st.markdown(f'<div class="section-header">Top {len(results)} Recommendations</div>', unsafe_allow_html=True)
                st.caption(f"⚡ Live Measured Latency: **{latency_ms:.1f} ms** across 287,421 papers | Ranked by fused lexical + topic alignment")

                for idx, r in enumerate(results, 1):
                    render_paper_card(r, rank=idx)

                # Score breakdown chart
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">⚖️ Score Contribution Breakdown</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-sub">Decomposition of hybrid score into TF-IDF and Topic arms</div>', unsafe_allow_html=True)
                fig_breakdown = plot_score_contributions(results)
                st.plotly_chart(fig_breakdown, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: SIMILAR PAPERS ("MORE LIKE THIS")
    # ══════════════════════════════════════════════════════════════════════════
    with tab_similar:
        col_s_left, col_s_right = st.columns([1.6, 1.0], gap="medium")

        with col_s_left:
            st.markdown('<div class="section-header">Select a Seed Paper</div>', unsafe_allow_html=True)
            seed_search = st.text_input("🔎 Search seed paper by title or arXiv ID:", placeholder="e.g. ResNet, Attention, Backtracking, BERT...", key="seed_search_input")

            if seed_search.strip():
                mask = (
                    catalog["title"].str.contains(seed_search, case=False, na=False)
                    | catalog["id"].str.contains(seed_search, case=False, na=False)
                )
                seed_candidates = catalog[mask].head(120).reset_index(drop=True)
            else:
                seed_candidates = catalog.sample(120, random_state=42).reset_index(drop=True)

            if seed_candidates.empty:
                st.warning("No papers match your search term. Try another title keyword.")
                return

            seed_options = [
                f"{r['title'][:60]}... ({r['id']})" if len(r['title']) > 60 else f"{r['title']} ({r['id']})"
                for _, r in seed_candidates.iterrows()
            ]
            seed_sel_idx = st.selectbox(
                "Choose seed paper from candidates:",
                range(len(seed_options)),
                format_func=lambda i: seed_options[i],
                key="seed_paper_select",
            )
            selected_seed = seed_candidates.iloc[seed_sel_idx]
            seed_id = selected_seed["id"]

        with col_s_right:
            st.markdown('<div class="section-header">Seed Paper Details</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px;">
                <span class="category-pill">📄 {selected_seed['id']}</span>
                <span class="category-pill">🏷️ {selected_seed['category']}</span>
                <div style="font-size:0.96rem;font-weight:600;color:#f0f6fc;margin:6px 0 4px;">{selected_seed['title']}</div>
                <div style="font-size:0.78rem;color:#8b949e;">✍️ {selected_seed['authors_display']} · 📅 {selected_seed['published_year']}</div>
            </div>""", unsafe_allow_html=True)

        # Similar papers weights & options
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        sw1, sw2, sw3 = st.columns([1, 1, 1])
        with sw1:
            w_sim_tfidf = st.slider("Lexical Match Weight", 0.0, 1.0, 0.60, 0.05, key="w_sim_tfidf")
        with sw2:
            w_sim_topic = st.slider("Topic Alignment Weight", 0.0, 1.0, 0.40, 0.05, key="w_sim_topic")
        with sw3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            constrain_cat = st.checkbox("Constrain to same category", value=False, key="constrain_cat_checkbox")
            sim_cat_filter = selected_seed["category"] if constrain_cat else None

        t0 = time.perf_counter()
        with st.spinner(f"Computing paper-to-paper similarities for {seed_id}…"):
            similar_results = hybrid.recommend_for_paper(
                paper_id=seed_id,
                top_k=top_k_setting,
                weight_tfidf=w_sim_tfidf,
                weight_topic=w_sim_topic,
                category=sim_cat_filter,
            )
        sim_latency_ms = (time.perf_counter() - t0) * 1000

        if not similar_results:
            st.info("No similar papers found matching the specified filter criteria.")
        else:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-header">Similar Research to <i>"{selected_seed["title"][:45]}..."</i></div>', unsafe_allow_html=True)
            st.caption(f"⚡ Live Measured Latency: **{sim_latency_ms:.1f} ms** across 287,421 papers | Source paper and all revision versions strictly excluded")

            for idx, r in enumerate(similar_results, 1):
                render_paper_card(r, rank=idx)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚖️ Score Composition</div>', unsafe_allow_html=True)
            fig_sim_breakdown = plot_score_contributions(similar_results)
            st.plotly_chart(fig_sim_breakdown, use_container_width=True)

