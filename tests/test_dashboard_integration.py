"""
Integration tests for LitLens Streamlit Dashboard.
Tests dashboard module imports, component rendering functions,
and cached model loader.
"""

import os
import unittest
import pandas as pd
from src.preprocessing import load_processed_catalog
from src.recommendation.hybrid_engine import HybridRecommender
from src.recommendation.topic_modeler import TopicModeler
from dashboard.components.styles import CUSTOM_CSS, apply_custom_css
from dashboard.components.charts import (
    plot_category_distribution,
    plot_publication_timeline,
    plot_score_contributions,
    plot_topic_overview,
)
from dashboard.components.cards import render_kpi_card


class TestDashboardIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models"
        cls.catalog = load_processed_catalog(
            os.path.join(cls.models_dir, "papers_catalog.parquet"),
            columns=["id", "base_id", "title", "category", "published_year", "authors_display", "first_author", "summary"],
        )
        cls.topic = TopicModeler()
        cls.topic.load_artifacts(cls.models_dir)
        cls.hybrid = HybridRecommender(models_dir=cls.models_dir, catalog=cls.catalog)

    def test_styles_and_kpi(self):
        self.assertIn("hero-banner", CUSTOM_CSS)
        self.assertIn("paper-card", CUSTOM_CSS)
        kpi_html = render_kpi_card("287,421", "Total Papers")
        self.assertIn("287,421", kpi_html)
        self.assertIn("Total Papers", kpi_html)

    def test_charts_generation(self):
        # Category distribution chart
        fig_cat = plot_category_distribution(self.catalog["category"].head(1000), top_n=5)
        self.assertIsNotNone(fig_cat)

        # Timeline chart
        fig_time = plot_publication_timeline(self.catalog["published_year"].head(1000))
        self.assertIsNotNone(fig_time)

        # Score contributions chart
        mock_recs = [
            {
                "title": "Attention Is All You Need",
                "tfidf_contribution": 0.45,
                "topic_contribution": 0.35,
            },
            {
                "title": "Deep Residual Learning",
                "tfidf_contribution": 0.40,
                "topic_contribution": 0.30,
            },
        ]
        fig_score = plot_score_contributions(mock_recs)
        self.assertIsNotNone(fig_score)

        # Topic overview chart
        topics = self.topic.get_all_topics()
        fig_topic = plot_topic_overview(topics, {0: 100, 1: 50})
        self.assertIsNotNone(fig_topic)

    def test_hybrid_engine_integrated_queries(self):
        # Verify that discovery query works with filters
        res = self.hybrid.recommend_for_query(
            "transformers self-attention",
            top_k=3,
            weight_tfidf=0.6,
            weight_topic=0.4,
            year_min=2017,
        )
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertGreater(r["final_score"], 0.0)
            self.assertGreaterEqual(r["published_year"], 2017)

    def test_paper_to_paper_integrated(self):
        sample_id = "cs-9308101v1"
        res = self.hybrid.recommend_for_paper(sample_id, top_k=3)
        self.assertEqual(len(res), 3)
        for r in res:
            self.assertNotEqual(r["paper_id"], sample_id)


if __name__ == "__main__":
    unittest.main()

