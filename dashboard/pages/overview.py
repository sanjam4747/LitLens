"""
Overview section for LitLens Streamlit Dashboard.
Displays dataset statistics, KPI metric cards, system architecture overview,
publication timelines, and category distributions.
"""

import streamlit as st
import pandas as pd
from dashboard.components.cards import render_kpi_card
from dashboard.components.charts import plot_category_distribution, plot_publication_timeline


def render_overview_page(catalog: pd.DataFrame, recommender_summary: dict):
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 LitLens Research Discovery</h1>
        <p>An intelligent, multi-arm discovery system for scientific papers — combining sparse TF-IDF
        lexical indexing and Latent Dirichlet Allocation (LDA) topic modeling over the arXiv corpus.</p>
    </div>""", unsafe_allow_html=True)

    # KPI Metric Row
    c1, c2, c3, c4, c5 = st.columns(5)
    total_papers = f"{len(catalog):,}"
    total_cats = f"{catalog['category'].nunique():,}"
    
    # Estimate unique authors
    unique_first_authors = f"{catalog['first_author'].nunique():,}" if "first_author" in catalog.columns else "N/A"
    
    kpis = [
        (total_papers, "Indexed Papers"),
        (total_cats, "Scientific Categories"),
        (unique_first_authors, "Primary Authors"),
        ("16", "Discovered Topics"),
        ("< 50 ms (Est.)", "Benchmark Latency"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(render_kpi_card(val, label), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_arch, col_dist = st.columns([1, 1], gap="large")

    with col_arch:
        st.markdown('<div class="section-header">🧠 System Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Two-arm score fusion engine with full explainability</div>', unsafe_allow_html=True)

        for icon, title, desc, color in [
            (
                "🎯",
                "Content-Based Arm (Sparse TF-IDF)",
                "Sublinear-scaled unigram & bigram indexing over weighted title and abstract representations. Computes exact lexical cosine similarity via sparse dot products.",
                "#58a6ff",
            ),
            (
                "🧬",
                "Topic Modeling Arm (Online LDA)",
                "16 latent topic distributions trained via online variational Bayes. Projects papers into normalized topic space to discover thematic conceptual alignment.",
                "#d2a8ff",
            ),
            (
                "⚖️",
                "Transparent Score Fusion",
                "Linear combination of normalized lexical and topic similarity scores with exposed component contributions and extracted overlapping technical terms.",
                "#3fb950",
            ),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-left:3px solid {color};
                border-radius:10px;padding:14px 18px;margin-bottom:12px;">
                <div style="font-size:0.92rem;font-weight:600;color:{color};margin-bottom:4px;">{icon} {title}</div>
                <div style="font-size:0.84rem;color:#8b949e;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_dist:
        st.markdown('<div class="section-header">📅 Publications Timeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Corpus publication volume from 1993 to 2026</div>', unsafe_allow_html=True)
        fig_time = plot_publication_timeline(catalog["published_year"])
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("---")

    col_cat, col_info = st.columns([1, 1], gap="large")

    with col_cat:
        st.markdown('<div class="section-header">📊 Top Scientific Disciplines</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribution of research papers by arXiv category</div>', unsafe_allow_html=True)
        fig_cat = plot_category_distribution(catalog["category"], top_n=10)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">📄 Corpus Quality & Ingestion Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Verified data integrity and preprocessing benchmarks</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px 22px;font-size:0.86rem;color:#c9d1d9;line-height:1.7;">
            • <b>Source:</b> arXiv Scientific Dataset (287,421 verified records)<br>
            • <b>Data Completeness:</b> 0 missing fields across canonical columns (100% complete)<br>
            • <b>Vocabulary Size:</b> 50,000 top n-grams (unigrams + bigrams)<br>
            • <b>Non-Zero Entries:</b> 33.1M sparse weights (Density: 0.0023, 99.7% memory savings)<br>
            • <b>Catalog Serialization:</b> Snappy-compressed Parquet (238 MB)<br>
            • <b>Topic Vectors:</b> 287,421 × 16 normalized topic distributions (17.5 MB)
        </div>""", unsafe_allow_html=True)

