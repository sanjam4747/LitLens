"""
Paper card and UI component rendering for LitLens.
Renders academic paper cards, KPI metric widgets, and arXiv links.
"""

from typing import Dict, List, Optional, Any
import streamlit as st


def render_kpi_card(val: str, label: str) -> str:
    """Returns HTML for a responsive KPI metric card."""
    return f"""
    <div class="metric-card">
        <span class="metric-value">{val}</span>
        <span class="metric-label">{label}</span>
    </div>"""


def render_paper_card(item: Dict[str, Any], rank: Optional[int] = None) -> None:
    """
    Renders an academic research paper card in Streamlit with badges,
    scores, authors, shared concepts, and an expandable abstract.
    """
    pid = item.get("paper_id", "")
    title = item.get("title", "Untitled")
    authors = item.get("authors", "Unknown Authors")
    year = item.get("published_year", "N/A")
    category = item.get("category", "")
    final_score = item.get("final_score", item.get("similarity_score", 0.0))
    tfidf_score = item.get("tfidf_score", None)
    topic_score = item.get("topic_score", None)
    dom_topic = item.get("dominant_topic", None)
    overlapping = item.get("overlapping_terms", [])
    summary = item.get("summary", "")

    # Clean arXiv link
    arxiv_id = pid
    if arxiv_id.startswith("abs-"):
        arxiv_id = arxiv_id[4:]
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"

    # Build badge HTML
    rank_html = f'<span class="rank-pill">#{rank}</span> ' if rank else ""
    cat_html = f'<span class="category-pill">🏷️ {category}</span>' if category else ""
    arxiv_badge = f'<a href="{arxiv_url}" target="_blank" style="text-decoration:none;"><span class="category-pill" style="color:#58a6ff;border-color:#1f6feb;">📄 arXiv:{pid}</span></a>'

    score_badges = f'<span class="score-badge-hybrid">🎯 Relevance: {final_score:.4f}</span>'
    if tfidf_score is not None:
        score_badges += f'<span class="score-badge-tfidf">🔍 Lexical: {tfidf_score:.4f}</span>'
    if topic_score is not None:
        score_badges += f'<span class="score-badge-topic">🧬 Topic: {topic_score:.4f}</span>'

    # Concept tags HTML
    concepts_html = ""
    if overlapping:
        concepts_html = "<div style='margin-top:8px;'><span style='font-size:0.75rem;color:#8b949e;margin-right:6px;'>Key Shared Concepts:</span>"
        for term, weight in overlapping[:4]:
            concepts_html += f'<span class="concept-tag">{term}</span>'
        concepts_html += "</div>"

    topic_info_html = ""
    if dom_topic:
        topic_info_html = f'<div style="font-size:0.78rem;color:#d2a8ff;margin-top:6px;">💡 Topic Alignment: <b>{dom_topic}</b></div>'

    card_html = f"""
    <div class="paper-card">
        <div class="paper-card-header">
            <div>
                {rank_html}{cat_html}{arxiv_badge}
                <div class="paper-title" style="margin-top:8px;">{title}</div>
            </div>
        </div>
        <div class="paper-meta">
            ✍️ <b>Authors:</b> {authors} &nbsp;·&nbsp; 📅 <b>Year:</b> {year}
        </div>
        <div style="margin-bottom:8px;">
            {score_badges}
        </div>
        {topic_info_html}
        {concepts_html}
    </div>"""

    st.markdown(card_html, unsafe_allow_html=True)
    if summary:
        with st.expander(f"📖 Abstract Preview ({pid})"):
            st.markdown(f'<div style="font-size:0.88rem;color:#c9d1d9;line-height:1.6;">{summary}</div>', unsafe_allow_html=True)

