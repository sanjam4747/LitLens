"""
Frontend verification script for LitLens.
Tests all three consolidated pages:
1. Overview: KPIs, architecture, timeline
2. Research Discovery: Query search, dynamic weight ranking shifts, and Similar Papers base_id version exclusion
3. Research Insights: 16 LDA topics, keyword terms, and topic-based paper discovery
"""

import sys
import os
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_processed_catalog
from src.recommendation.tfidf_recommender import TFIDFRecommender
from src.recommendation.topic_modeler import TopicModeler
from src.recommendation.hybrid_engine import HybridRecommender
from dashboard.components.charts import (
    plot_category_distribution,
    plot_publication_timeline,
    plot_score_contributions,
    plot_topic_overview,
)


def main():
    print("=" * 70)
    print("  LitLens Frontend Refactoring Verification")
    print("=" * 70)

    # 1. Load artifacts
    models_dir = os.path.join(PROJECT_ROOT, "models")
    catalog = load_processed_catalog(os.path.join(models_dir, "papers_catalog.parquet"))
    tfidf = TFIDFRecommender()
    tfidf.load_artifacts(models_dir)
    topic = TopicModeler()
    topic.load_artifacts(models_dir)
    hybrid = HybridRecommender(tfidf_recommender=tfidf, topic_modeler=topic, catalog=catalog, models_dir=models_dir)

    print("\n[VERIFICATION 1] Page 1: Overview KPIs & Charts")
    print(f"  Indexed Papers: {len(catalog):,}")
    print(f"  Scientific Categories: {catalog['category'].nunique()}")
    print(f"  Discovered Topics: {topic.n_components}")
    fig_time = plot_publication_timeline(catalog["published_year"])
    fig_cat = plot_category_distribution(catalog["category"], top_n=10)
    assert fig_time is not None and fig_cat is not None
    print("  Overview charts successfully generated: PASS")

    print("\n[VERIFICATION 2] Page 2: Research Discovery (Natural-Language Search)")
    q = "deep residual networks for image recognition and feature extraction"
    res_query = hybrid.recommend_for_query(q, top_k=5, weight_tfidf=0.6, weight_topic=0.4)
    print(f"  Query: '{q}' -> Returned {len(res_query)} recommendations:")
    for idx, r in enumerate(res_query, 1):
        print(f"    #{idx} [{r['final_score']:.4f}] {r['paper_id']} - {r['title'][:48]}...")
    assert len(res_query) == 5
    print("  Natural-language query recommendation: PASS")

    print("\n[VERIFICATION 3] Page 2: Dynamic Ranking Shift on Weight Tuning")
    r_lex = hybrid.recommend_for_query(q, top_k=5, weight_tfidf=1.0, weight_topic=0.0)
    r_hyb = hybrid.recommend_for_query(q, top_k=5, weight_tfidf=0.5, weight_topic=0.5)
    r_top = hybrid.recommend_for_query(q, top_k=5, weight_tfidf=0.1, weight_topic=0.9)
    lex_ids = [x["paper_id"] for x in r_lex]
    hyb_ids = [x["paper_id"] for x in r_hyb]
    top_ids = [x["paper_id"] for x in r_top]
    print(f"  Lexical top-5 (1.0, 0.0): {lex_ids}")
    print(f"  Hybrid top-5  (0.5, 0.5): {hyb_ids}")
    print(f"  Topic top-5   (0.1, 0.9): {top_ids}")
    assert lex_ids != hyb_ids or hyb_ids != top_ids
    print("  Dynamic weight-based rank shifts: PASS")

    print("\n[VERIFICATION 4] Page 2: Similar Papers with Base ID Version Exclusion")
    seed_id = "abs-2501.14993v1"
    seed_base_id = "abs-2501.14993"
    sim_recs = hybrid.recommend_for_paper(seed_id, top_k=5)
    print(f"  Seed Paper: {seed_id} (Base ID: {seed_base_id})")
    print(f"  Returned {len(sim_recs)} similar papers:")
    for idx, r in enumerate(sim_recs, 1):
        pid = r["paper_id"]
        print(f"    #{idx} [{r['final_score']:.4f}] {pid} - {r['title'][:48]}...")
        assert not pid.startswith(seed_base_id), f"Leak: {pid} matches seed base_id {seed_base_id}"
    print("  Version deduplication & seed exclusion: PASS")

    print("\n[VERIFICATION 5] Page 3: Research Insights (LDA Topics & Paper Discovery)")
    topics = topic.get_all_topics()
    print(f"  Total Discovered Topics: {len(topics)}")
    assert len(topics) == 16
    for tid in [0, 1, 3, 6, 8]:
        t_info = topics[tid]
        kw = ", ".join([w[0] for w in t_info["top_terms"][:4]])
        print(f"    Topic {tid+1:02d} [{t_info['label']}]: {kw}")

    disc_papers = topic.discover_papers_by_topic(0, top_k=3)
    print(f"  Discovered {len(disc_papers)} papers for Topic 01:")
    for pid, w in disc_papers:
        print(f"    Paper: {pid} (Topic Alignment Weight: {w:.4f})")
    assert len(disc_papers) == 3
    print("  Topic discovery & exploration: PASS")

    print("\n" + "=" * 70)
    print("  All Frontend Refactoring Verifications PASSED Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

