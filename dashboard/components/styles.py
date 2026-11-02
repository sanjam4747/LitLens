"""
Custom CSS styling and theming for LitLens Streamlit Dashboard.
Academic research-product aesthetic: clean dark mode, responsive layout,
compact padding, refined typography, and explicit suppression of duplicate nav.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Ensure Streamlit's auto-generated multi-page sidebar navigation is hidden */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Tighten default Streamlit page padding and remove dead space */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1350px !important;
}

.main {
    background-color: #0d1117;
    color: #e6edf3;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
    padding-top: 1rem;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0c2340 0%, #0d3868 45%, #184e77 80%, #1a535c 100%);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    border: 1px solid #1f4260;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 180px;
    height: 180px;
    background: rgba(255, 255, 255, 0.035);
    border-radius: 50%;
}
.hero-banner h1 {
    color: #ffffff !important;
    font-size: 1.95rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px !important;
    letter-spacing: -0.4px;
}
.hero-banner p {
    color: rgba(255, 255, 255, 0.88) !important;
    font-size: 0.96rem !important;
    margin: 0 !important;
    font-weight: 400;
    line-height: 1.45;
}

/* Metric KPI Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    height: 100%;
}
.metric-card:hover {
    border-color: #58a6ff;
    transform: translateY(-1px);
}
.metric-card .metric-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #58a6ff;
    display: block;
    line-height: 1.15;
}
.metric-card .metric-label {
    font-size: 0.74rem;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 4px;
}

/* Paper Result Cards */
.paper-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.paper-card:hover {
    border-color: #58a6ff;
    box-shadow: 0 3px 14px rgba(88, 166, 255, 0.10);
}
.paper-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 10px;
}
.paper-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f0f6fc;
    line-height: 1.35;
    margin-top: 6px;
    margin-bottom: 6px;
}
.paper-meta {
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 10px;
    line-height: 1.4;
}
.paper-meta b {
    color: #c9d1d9;
}
.paper-abstract {
    font-size: 0.85rem;
    color: #c9d1d9;
    line-height: 1.5;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 12px;
    margin: 8px 0;
}

/* Badges and Tag Chips */
.rank-pill {
    display: inline-block;
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
}
.category-pill {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    color: #9ca3af;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 5px;
    margin-right: 5px;
}
.score-badge-hybrid {
    display: inline-block;
    background: linear-gradient(90deg, #1f6feb, #238636);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 16px;
    margin-right: 5px;
}
.score-badge-tfidf {
    display: inline-block;
    background: rgba(88, 166, 255, 0.12);
    border: 1px solid rgba(88, 166, 255, 0.35);
    color: #79c0ff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
    margin-right: 5px;
}
.score-badge-topic {
    display: inline-block;
    background: rgba(210, 168, 255, 0.12);
    border: 1px solid rgba(210, 168, 255, 0.35);
    color: #d2a8ff;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 10px;
    margin-right: 5px;
}
.concept-tag {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    font-size: 0.70rem;
    padding: 2px 7px;
    border-radius: 8px;
    margin-right: 4px;
    margin-top: 3px;
}

/* Headings and Hierarchy */
.section-header {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-top: 6px;
    margin-bottom: 2px;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}
.section-sub {
    font-size: 0.82rem;
    color: #8b949e;
    margin-bottom: 12px;
}
.info-box {
    background: rgba(88, 166, 255, 0.07);
    border: 1px solid rgba(88, 166, 255, 0.22);
    border-radius: 8px;
    padding: 10px 14px;
    color: #8b949e;
    font-size: 0.82rem;
    margin: 8px 0;
}
.nav-title {
    font-size: 0.70rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 14px;
    margin-bottom: 6px;
    padding-left: 2px;
}

/* Styled Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 14px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border: none;
    color: #8b949e;
    font-weight: 500;
    font-size: 0.90rem;
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background-color: #161b22 !important;
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    font-weight: 600;
}
</style>
"""


def apply_custom_css():
    """Injects custom CSS styles into the Streamlit session."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
