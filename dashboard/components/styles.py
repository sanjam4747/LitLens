"""
Custom CSS styling and theming for LitLens Streamlit Dashboard.
Adapts BookMind dark theme patterns for an academic research discovery platform.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.main {
    background-color: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0d2137 0%, #0d3b66 40%, #1b4965 70%, #2a6f97 100%);
    border-radius: 14px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    border: 1px solid #284b63;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px;
    right: -60px;
    width: 220px;
    height: 220px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 50%;
}
.hero-banner h1 {
    color: #ffffff !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin: 0 0 8px !important;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255, 255, 255, 0.85) !important;
    font-size: 1.02rem !important;
    margin: 0 !important;
    font-weight: 400;
}

/* Metric KPI Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    height: 100%;
}
.metric-card:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
}
.metric-card .metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #58a6ff;
    display: block;
    line-height: 1.2;
}
.metric-card .metric-label {
    font-size: 0.80rem;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 6px;
}

/* Paper Cards */
.paper-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}
.paper-card:hover {
    border-color: #58a6ff;
    box-shadow: 0 4px 20px rgba(88, 166, 255, 0.12);
    transform: translateY(-2px);
}
.paper-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 8px;
}
.paper-title {
    font-size: 1.12rem;
    font-weight: 600;
    color: #f0f6fc;
    line-height: 1.4;
}
.paper-meta {
    font-size: 0.84rem;
    color: #8b949e;
    margin-bottom: 12px;
}
.paper-meta b {
    color: #c9d1d9;
}
.paper-abstract {
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.55;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 12px 0;
}

/* Badges and Tags */
.rank-pill {
    display: inline-block;
    background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff;
    font-size: 0.76rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 14px;
}
.category-pill {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    color: #9ca3af;
    font-size: 0.74rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 6px;
    margin-right: 6px;
}
.score-badge-hybrid {
    display: inline-block;
    background: linear-gradient(90deg, #1f6feb, #238636);
    color: #ffffff;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
}
.score-badge-tfidf {
    display: inline-block;
    background: rgba(88, 166, 255, 0.15);
    border: 1px solid rgba(88, 166, 255, 0.4);
    color: #79c0ff;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 6px;
}
.score-badge-topic {
    display: inline-block;
    background: rgba(210, 168, 255, 0.15);
    border: 1px solid rgba(210, 168, 255, 0.4);
    color: #d2a8ff;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 12px;
    margin-right: 6px;
}
.concept-tag {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    font-size: 0.73rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin-right: 4px;
    margin-top: 4px;
}

/* Headers & Sections */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f0f6fc;
    margin-bottom: 4px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}
.section-sub {
    font-size: 0.86rem;
    color: #8b949e;
    margin-bottom: 18px;
}
.info-box {
    background: rgba(88, 166, 255, 0.08);
    border: 1px solid rgba(88, 166, 255, 0.25);
    border-radius: 8px;
    padding: 12px 16px;
    color: #8b949e;
    font-size: 0.84rem;
    margin: 10px 0;
}
.nav-title {
    font-size: 0.72rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 18px;
    margin-bottom: 8px;
    padding-left: 2px;
}
</style>
"""


def apply_custom_css():
    """Injects custom CSS styles into the Streamlit session."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

