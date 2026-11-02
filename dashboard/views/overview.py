"""
Page 1: Overview View for LitLens Streamlit Dashboard.
Displays project introduction, dataset KPIs, system architecture cards,
publication timeline, top scientific categories, and dataset quality summary.
"""

import streamlit as st
import pandas as pd
from dashboard.components.cards import render_kpi_card
from dashboard.components.charts import plot_category_distribution, plot_publication_timeline


def render_overview_page(catalog: pd.DataFrame, recommender_summary: dict):
    st.markdown("""
    <div class="hero-banner">
        <h1>🔬 LitLens — Scientific Paper Discovery System</h1>
        <p>An intelligent, interpretable discovery platform indexing 287,421 arXiv research papers.
        Combines sparse TF-IDF lexical matching with 16-topic Latent Dirichlet Allocation (LDA) for
        transparent, multi-arm paper exploration.</p>
    </div>""", unsafe_allow_html=True)

    # KPI Metric Row (Compact, responsive 5 columns)
    c1, c2, c3, c4, c5 = st.columns(5)
    total_papers = f"{len(catalog):,}"
    total_cats = f"{catalog['category'].nunique():,}"
    unique_first_authors = f"{catalog['first_author'].nunique():,}" if "first_author" in catalog.columns else "N/A"

    kpis = [
        (total_papers, "Indexed Papers"),
        (total_cats, "Categories"),
        ("16", "Discovered Topics"),
        ("50,000", "N-Gram Vocabulary"),
        ("< 50 ms (Est.)", "Benchmark Latency"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(render_kpi_card(val, label), unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Row 1: System Architecture (Left) + Publication Timeline (Right)
    col_arch, col_time = st.columns([1, 1], gap="medium")

    with col_arch:
        st.markdown('<div class="section-header">🧠 System Architecture</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Two-arm score fusion engine with full explainability</div>', unsafe_allow_html=True)

        for icon, title, desc, color in [
            (
                "🎯",
                "Content-Based Arm (Sparse TF-IDF)",
                "Sublinear-scaled unigram & bigram indexing over weighted title (3×) and abstract text. Computes exact lexical cosine similarity via sparse dot products.",
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
                "Linear combination of normalized lexical and topic similarity scores: Final = w₁·TFIDF + w₂·Topic. Exposes component contributions and shared concepts.",
                "#3fb950",
            ),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-left:3px solid {color};
                border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                <div style="font-size:0.90rem;font-weight:600;color:{color};margin-bottom:3px;">{icon} {title}</div>
                <div style="font-size:0.82rem;color:#8b949e;line-height:1.45;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col_time:
        st.markdown('<div class="section-header">📅 Publications Timeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Corpus volume growth from 1993 to 2026</div>', unsafe_allow_html=True)
        fig_time = plot_publication_timeline(catalog["published_year"])
        st.plotly_chart(fig_time, use_container_width=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Row 2: Top Disciplines (Left) + Dataset Quality Summary (Right)
    col_cat, col_info = st.columns([1, 1], gap="medium")

    with col_cat:
        st.markdown('<div class="section-header">📊 Top Scientific Disciplines</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Distribution of research papers by arXiv category</div>', unsafe_allow_html=True)
        fig_cat = plot_category_distribution(catalog["category"], top_n=10)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_info:
        st.markdown('<div class="section-header">📄 Dataset Quality & Ingestion Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Verified data integrity and preprocessing benchmarks</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px 20px;font-size:0.84rem;color:#c9d1d9;line-height:1.75;">
            • <b>Raw Corpus:</b> arXiv Scientific Dataset (287,421 verified records)<br>
            • <b>Data Integrity:</b> 0 missing fields across all 10 canonical columns (100% complete)<br>
            • <b>Primary Identifiers:</b> 287,421 unique IDs · 287,395 unique base papers<br>
            • <b>Vocabulary Size:</b> 50,000 top n-grams (unigrams + bigrams)<br>
            • <b>Sparse Matrix Density:</b> 0.002304 (33.1M non-zero entries, 99.7% RAM savings)<br>
            • <b>Catalog Cache:</b> Columnar Parquet with Snappy compression (238 MB)<br>
            • <b>Topic Space:</b> 287,421 × 16 normalized topic vectors (17.5 MB)
        </div>""", unsafe_allow_html=True)

