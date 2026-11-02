"""
Analytics and Dataset Insights page for LitLens Streamlit Dashboard.
Adapts BookMind analytics layout for scientific publication patterns,
category distributions, and model weight dynamics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dashboard.components.charts import (
    CHART_LAYOUT,
    COLORS,
    plot_category_distribution,
    plot_publication_timeline,
    plot_topic_overview,
)
from src.recommendation.topic_modeler import TopicModeler


def render_analytics_page(catalog: pd.DataFrame, topic_modeler: TopicModeler):
    st.markdown("""
    <div class="hero-banner" style="background:linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);">
        <h1>📊 Corpus Analytics & Methodology</h1>
        <p>In-depth analytical exploration of the 287k arXiv research papers — category breakdowns,
        temporal publication trends, and NLP model properties.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📚 Categories & Fields", "📅 Publication Trends", "🧬 Topic Dynamics"])

    with tab1:
        c1, c2 = st.columns([1, 1], gap="large")
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
                margin=dict(t=20, b=20, l=20, r=20),
                height=380,
                legend=dict(bgcolor="#161b22", font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        c3, c4 = st.columns([1, 1], gap="large")
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
            fig_dec.update_layout(**CHART_LAYOUT, height=340, xaxis_title="Decade", yaxis_title="Papers Count")
            st.plotly_chart(fig_dec, use_container_width=True)

        with c4:
            st.markdown('<div class="section-header">📈 Annual Research Growth Timeline</div>', unsafe_allow_html=True)
            fig_growth = plot_publication_timeline(catalog["published_year"])
            st.plotly_chart(fig_growth, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">🧬 LDA Topic Representation Across All Papers</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">16 unsupervised topics discovered using variational inference</div>', unsafe_allow_html=True)

        topics = topic_modeler.get_all_topics()
        dominant_counts = {}
        if topic_modeler.topic_distributions is not None:
            dominant_indices = topic_modeler.topic_distributions.argmax(axis=1)
            u, counts = np.unique(dominant_indices, return_counts=True)
            dominant_counts = dict(zip([int(x) for x in u], [int(y) for y in counts]))

        fig_top = plot_topic_overview(topics, dominant_counts)
        st.plotly_chart(fig_top, use_container_width=True)

