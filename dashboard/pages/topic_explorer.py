"""
Topic Explorer page for LitLens Streamlit Dashboard.
Visualizes the 16 Latent Dirichlet Allocation (LDA) topics, representative terms,
and discovers top representative papers per topic cluster.
"""

from typing import Dict, Any
import numpy as np
import streamlit as st
import pandas as pd
from dashboard.components.cards import render_paper_card
from dashboard.components.charts import plot_topic_overview
from src.recommendation.topic_modeler import TopicModeler


def render_topic_explorer_page(topic_modeler: TopicModeler, catalog: pd.DataFrame, top_k_setting: int):
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg, #4c1d95 0%, #6d28d9 50%, #8b5cf6 100%);">
        <h1>🧬 Scientific Topic Explorer</h1>
        <p>Explore 16 unsupervised semantic topics discovered across 287k papers using online LDA.
        Discover core research clusters and top representative papers.</p>
    </div>""", unsafe_allow_html=True)

    topics = topic_modeler.get_all_topics()

    # Compute dominant topic counts across corpus
    if topic_modeler.topic_distributions is not None:
        dominant_indices = topic_modeler.topic_distributions.argmax(axis=1)
        unique, counts = np.unique(dominant_indices, return_counts=True)
        dominant_counts = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
    else:
        dominant_counts = {k: 0 for k in topics.keys()}

    col_chart, col_topics = st.columns([1, 1], gap="large")

    with col_chart:
        st.markdown('<div class="section-header">📊 Topic Distribution Across arXiv</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Volume of papers predominantly aligned with each topic</div>', unsafe_allow_html=True)
        fig_topics = plot_topic_overview(topics, dominant_counts)
        st.plotly_chart(fig_topics, use_container_width=True)

    with col_topics:
        st.markdown('<div class="section-header">🔍 Inspect Topic Keywords</div>', unsafe_allow_html=True)
        topic_options = [f"Topic {k+1:02d}: {t['label']}" for k, t in topics.items()]
        selected_idx = st.selectbox(
            "Select a topic cluster to explore:",
            range(len(topic_options)),
            format_func=lambda i: topic_options[i],
        )

        sel_topic_info = topics[selected_idx]
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px 22px;margin-bottom:16px;">
            <div style="font-size:1.15rem;font-weight:700;color:#d2a8ff;">Topic {selected_idx+1:02d}: {sel_topic_info['label']}</div>
            <div style="font-size:0.82rem;color:#8b949e;margin-top:4px;">Volume: ~{dominant_counts.get(selected_idx, 0):,} papers</div>
            <div style="margin-top:12px;font-size:0.86rem;color:#e6edf3;"><b>Representative Scientific Terms:</b></div>
            <div style="margin-top:8px;">
        """, unsafe_allow_html=True)

        for term, weight in sel_topic_info["top_terms"]:
            st.markdown(f'<span class="concept-tag" style="border-color:#8957e5;color:#d2a8ff;">{term} ({weight:.3f})</span>', unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Topic-based paper discovery
    st.markdown("---")
    st.markdown(f'<div class="section-header">🏆 Top Representative Papers for Topic {selected_idx+1:02d}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Papers exhibiting the highest statistical alignment with this topic</div>', unsafe_allow_html=True)

    discovered = topic_modeler.discover_papers_by_topic(selected_idx, top_k=top_k_setting)
    catalog_indexed = catalog.set_index("id", drop=False)

    for rank, (pid, weight) in enumerate(discovered, 1):
        if pid in catalog_indexed.index:
            row = catalog_indexed.loc[pid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            item = {
                "paper_id": pid,
                "title": row["title"],
                "category": row["category"],
                "published_year": int(row["published_year"]),
                "authors": row["authors_display"],
                "similarity_score": weight,
                "final_score": weight,
                "dominant_topic": sel_topic_info["label"],
                "summary": row["summary"],
                "explanation": f"Statistical alignment of {weight:.4f} with Topic {selected_idx+1:02d} ({sel_topic_info['label']}).",
            }
            render_paper_card(item, rank=rank)
