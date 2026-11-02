"""
LitLens — NLP-Based Intelligent Research Paper Recommendation and Discovery System
Main Streamlit Application Entrypoint.
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
from dashboard.pages.overview import render_overview_page
from dashboard.pages.discovery import render_discovery_page
from dashboard.pages.similar_papers import render_similar_papers_page
from dashboard.pages.topic_explorer import render_topic_explorer_page
from dashboard.pages.analytics import render_analytics_page

from src.preprocessing import load_processed_catalog
from src.recommendation.tfidf_recommender import TFIDFRecommender
from src.recommendation.topic_modeler import TopicModeler
from src.recommendation.hybrid_engine import HybridRecommender

# Set page configuration
st.set_page_config(
    page_title="LitLens — Research Paper Recommendation System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark theme styling
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
# SIDEBAR NAVIGATION & SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 16px;">
        <div style="font-size:2.4rem;">🔬</div>
        <div style="font-size:1.3rem;font-weight:700;color:#f0f6fc;letter-spacing:-0.5px;">LitLens</div>
        <div style="font-size:0.78rem;color:#8b949e;">Research Paper Discovery</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Research Discovery",
            "📖 Similar Papers",
            "🧬 Topic Explorer",
            "📊 Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="nav-title">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Max Results to Display", min_value=5, max_value=30, value=10, step=5)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#8b949e;text-align:center;line-height:1.7;">
        <b style="color:#e6edf3;">Models</b><br>
        Sparse TF-IDF (50k n-grams)<br>
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

elif page == "📖 Similar Papers":
    render_similar_papers_page(hybrid_engine, catalog_df, top_k)

elif page == "🧬 Topic Explorer":
    render_topic_explorer_page(topic_engine, catalog_df, top_k)

elif page == "📊 Analytics":
    render_analytics_page(catalog_df, topic_engine)

