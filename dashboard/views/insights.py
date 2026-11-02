"""
Page 3: Research Insights View for LitLens Streamlit Dashboard.
Integrates:
- Tab 1: LDA Topic Explorer & Topic-Based Paper Discovery
- Tab 2: Corpus Analytics, Category Distributions & Publication Trends
"""

from typing import Dict, Any
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dashboard.components.cards import render_paper_card
from dashboard.components.charts import (
    CHART_LAYOUT,
    COLORS,
    plot_category_distribution,
    plot_publication_timeline,
    plot_topic_overview,
)
from src.recommendation.topic_modeler import TopicModeler


def render_insights_page(catalog: pd.DataFrame, topic_modeler: TopicModeler, top_k_setting: int):
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg, #3b0764 0%, #581c87 50%, #7e22ce 100%);">
        <h1>📊 Research Insights & Topic Dynamics</h1>
        <p>Analyze semantic themes across 287k papers. Explore 16 unsupervised topics discovered via online
        variational LDA and inspect publication trends and category distributions.</p>
    </div>""", unsafe_allow_html=True)

    tab_topics, tab_analytics = st.tabs(["🧬 Topic Explorer & Paper Discovery", "📈 Corpus & Publication Trends"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: TOPIC EXPLORER & PAPER DISCOVERY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_topics:
        topics = topic_modeler.get_all_topics()

        # Compute dominant topic counts across corpus
        if topic_modeler.topic_distributions is not None:
            dominant_indices = topic_modeler.topic_distributions.argmax(axis=1)
            u, counts = np.unique(dominant_indices, return_counts=True)
            dominant_counts = dict(zip([int(x) for x in u], [int(y) for y in counts]))
        else:
            dominant_counts = {k: 0 for k in topics.keys()}

        col_t_left, col_t_right = st.columns([1.1, 0.9], gap="medium")

        with col_t_left:
            st.markdown('<div class="section-header">📊 Topic Distribution Across arXiv</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">Volume of papers predominantly aligned with each topic</div>', unsafe_allow_html=True)
            fig_topics = plot_topic_overview(topics, dominant_counts)
            st.plotly_chart(fig_topics, use_container_width=True)

        with col_t_right:
            st.markdown('<div class="section-header">🔍 Inspect Topic Keywords</div>', unsafe_allow_html=True)
            topic_options = [f"Topic {k+1:02d}: {t['label']}" for k, t in topics.items()]
            selected_idx = st.selectbox(
                "Select a topic cluster to explore:",
                range(len(topic_options)),
                format_func=lambda i: topic_options[i],
                key="topic_explorer_select",
            )

            sel_topic_info = topics[selected_idx]
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px;margin-bottom:12px;">
                <div style="font-size:1.05rem;font-weight:700;color:#d2a8ff;">Topic {selected_idx+1:02d}: {sel_topic_info['label']}</div>
                <div style="font-size:0.78rem;color:#8b949e;margin-top:2px;">Volume: ~{dominant_counts.get(selected_idx, 0):,} papers</div>
                <div style="margin-top:10px;font-size:0.82rem;color:#e6edf3;"><b>Representative Scientific Terms:</b></div>
                <div style="margin-top:6px;">
            """, unsafe_allow_html=True)

            for term, weight in sel_topic_info["top_terms"]:
                st.markdown(f'<span class="concept-tag" style="border-color:#8957e5;color:#d2a8ff;">{term} ({weight:.3f})</span>', unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # Topic-based paper discovery
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">🏆 Top Representative Papers for Topic {selected_idx+1:02d}: <i>{sel_topic_info["label"]}</i></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Papers exhibiting the highest statistical alignment with this theme</div>', unsafe_allow_html=True)

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

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: CORPUS & PUBLICATION TRENDS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_analytics:
        c1, c2 = st.columns([1, 1], gap="medium")
        with c1:
            st.markdown('<div class="section-header">🏆 Top 15 Research Disciplines</div>', unsafe_allow_html=True)
            fig_cats = plot_category_distribution(catalog["category"], top_n=15)
            st.plotly_chart(fig_cats, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">🥧 Category Share (Top 6 + Others)</div>', unsafe_allow_html=True)
            top6 = catalog["category"].value_counts().head(6)
            other_count = len(catalog) - top6.sum()
            shares = pd.concat([top6, pd.Series({"Other Disciplines": other_count})])
            fig_pie = px.pie(
                values=shares.values,
                names=shares.index,
                hole=0.45,
                color_discrete_sequence=COLORS * 2,
            )
            fig_pie.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3", family="Inter"),
                margin=dict(t=15, b=15, l=15, r=15),
                height=380,
                legend=dict(bgcolor="#161b22", font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns([1, 1], gap="medium")
        with c3:
            st.markdown('<div class="section-header">📅 Papers Published Per Decade</div>', unsafe_allow_html=True)
            dec_df = catalog[catalog["published_year"] >= 1990].copy()
            dec_df["Decade"] = (dec_df["published_year"] // 10 * 10).astype(int)
            dec_counts = dec_df.groupby("Decade").size().reset_index(name="Count")
            fig_dec = go.Figure(go.Bar(
                x=[f"{d}s" for d in dec_counts["Decade"]],
                y=dec_counts["Count"],
                marker=dict(color=COLORS[0]),
                text=dec_counts["Count"],
                textposition="outside",
            ))
            fig_dec.update_layout(**CHART_LAYOUT, height=330, xaxis_title="Decade", yaxis_title="Papers Count")
            st.plotly_chart(fig_dec, use_container_width=True)

        with c4:
            st.markdown('<div class="section-header">📈 Annual Research Growth Timeline</div>', unsafe_allow_html=True)
            fig_growth = plot_publication_timeline(catalog["published_year"])
            st.plotly_chart(fig_growth, use_container_width=True)

