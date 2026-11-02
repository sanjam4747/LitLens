"""
LitLens — NLP-Based Intelligent Research Paper Recommendation and Discovery System
Main Streamlit Application Entrypoint.
Consolidated into THREE core views:
1. Overview (Project Introduction, Architecture, KPIs & Timeline)
2. Research Discovery (Natural-Language Search & Similar Papers)
3. Research Insights (LDA Topic Explorer & Corpus Analytics)
"""

import os
import sys
import warnings
import streamlit as st
import pandas as pd

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.components.styles import apply_custom_css
from dashboard.views.overview import render_overview_page
from dashboard.views.discovery import render_discovery_page
from dashboard.views.insights import render_insights_page

from src.preprocessing import load_processed_catalog
from src.recommendation.tfidf_recommender import TFIDFRecommender
from src.recommendation.topic_modeler import TopicModeler
from src.recommendation.hybrid_engine import HybridRecommender

# Set page configuration
st.set_page_config(
    page_title="LitLens — Research Paper Discovery System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme styling (suppresses default multi-page nav and fixes spacing)
apply_custom_css()

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@st.cache_resource(show_spinner="Loading NLP and recommendation models…")
def load_recommendation_system():
    """Cached loader for recommendation engine and topic modeler."""
    tfidf = TFIDFRecommender()
    tfidf.load_artifacts(MODELS_DIR)

    topic = TopicModeler()
    topic.load_artifacts(MODELS_DIR)

    catalog = load_processed_catalog(
        os.path.join(MODELS_DIR, "papers_catalog.parquet"),
        columns=[
            "id",
            "base_id",
            "title",
            "category",
            "published_year",
            "authors_display",
            "first_author",
            "summary",
        ],
    )

    hybrid = HybridRecommender(
        tfidf_recommender=tfidf,
        topic_modeler=topic,
        catalog=catalog,
        models_dir=MODELS_DIR,
    )

    return tfidf, topic, catalog, hybrid


# Load cached resources
tfidf_engine, topic_engine, catalog_df, hybrid_engine = load_recommendation_system()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION & SETTINGS (SINGLE NAVIGATION SYSTEM)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 14px;">
        <div style="font-size:2.2rem;line-height:1;">🔬</div>
        <div style="font-size:1.25rem;font-weight:700;color:#f0f6fc;letter-spacing:-0.4px;margin-top:4px;">LitLens</div>
        <div style="font-size:0.75rem;color:#8b949e;">Research Paper Discovery</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Research Discovery",
            "📊 Research Insights",
        ],
        label_visibility="collapsed",
        key="litlens_main_nav",
    )

    st.markdown('<div class="nav-title">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Max Results to Display", min_value=5, max_value=30, value=10, step=5, key="global_top_k")

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.74rem;color:#8b949e;text-align:center;line-height:1.65;">
        <b style="color:#e6edf3;">Active Models</b><br>
        Sparse TF-IDF (50k N-Grams)<br>
        Online LDA (16 Topics)<br><br>
        <b style="color:#e6edf3;">Dataset</b><br>
        arXiv Scientific Corpus<br>
        287,421 Indexed Papers
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    summary_info = {
        "total_records": len(catalog_df),
        "total_categories": catalog_df["category"].nunique(),
    }
    render_overview_page(catalog_df, summary_info)

elif page == "🔍 Research Discovery":
    render_discovery_page(hybrid_engine, catalog_df, top_k)

elif page == "📊 Research Insights":
    render_insights_page(catalog_df, topic_engine, top_k)
