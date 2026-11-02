"""
Unit tests for HybridRecommender engine.
Tests score fusion, independent score exposure, duplicate version exclusion,
category and year filtering, ranking stability, and lexical vs hybrid comparison.
"""

import unittest
from src.recommendation.hybrid_engine import HybridRecommender


class TestHybridRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.recommender = HybridRecommender(models_dir="models")

    def test_recommend_for_paper_basic(self):
        sample_id = "cs-9308101v1"
        results = self.recommender.recommend_for_paper(sample_id, top_k=5, weight_tfidf=0.6, weight_topic=0.4)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("paper_id", r)
            self.assertIn("title", r)
            self.assertIn("final_score", r)
            self.assertIn("tfidf_score", r)
            self.assertIn("topic_score", r)
            self.assertIn("dominant_topic", r)
            self.assertIn("explanation", r)
            self.assertGreater(r["final_score"], 0.0)
            self.assertLessEqual(r["final_score"], 1.0)

    def test_duplicate_version_exclusion(self):
        # abs-2501.14993 has versions v1 and v3 in dataset (base_id: abs-2501.14993)
        sample_id = "abs-2501.14993v1"
        target_base_id = "abs-2501.14993"
        results = self.recommender.recommend_for_paper(sample_id, top_k=10)

        # Ensure no paper with the same base_id appears in results
        for r in results:
            pid = r["paper_id"]
            self.assertFalse(
                pid.startswith(target_base_id),
                f"Version duplicate '{pid}' found in recommendations for source '{sample_id}'"
            )

    def test_category_filter(self):
        sample_id = "cs-9308101v1"
        target_category = "Artificial Intelligence"
        results = self.recommender.recommend_for_paper(
            sample_id,
            top_k=5,
            category=target_category,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["category"], target_category)

    def test_publication_year_filter(self):
        sample_id = "cs-9308101v1"
        year_min = 2018
        year_max = 2022
        results = self.recommender.recommend_for_paper(
            sample_id,
            top_k=5,
            year_min=year_min,
            year_max=year_max,
        )
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertGreaterEqual(r["published_year"], year_min)
            self.assertLessEqual(r["published_year"], year_max)

    def test_hybrid_score_calculation(self):
        sample_id = "cs-9308101v1"
        w_tfidf = 0.7
        w_topic = 0.3
        results = self.recommender.recommend_for_paper(
            sample_id,
            top_k=5,
            weight_tfidf=w_tfidf,
            weight_topic=w_topic,
        )
        for r in results:
            expected = round(w_tfidf * r["tfidf_score"] + w_topic * r["topic_score"], 4)
            self.assertAlmostEqual(r["final_score"], expected, places=2)
            self.assertAlmostEqual(r["tfidf_contribution"], w_tfidf * r["tfidf_score"], places=2)
            self.assertAlmostEqual(r["topic_contribution"], w_topic * r["topic_score"], places=2)

    def test_recommend_for_query(self):
        query = "graph neural network node classification representation learning"
        results = self.recommender.recommend_for_query(query, top_k=5, weight_tfidf=0.5, weight_topic=0.5)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertGreater(r["final_score"], 0.0)
            self.assertLessEqual(r["final_score"], 1.0)
            self.assertIn("explanation", r)

    def test_empty_and_invalid_queries(self):
        # Empty query
        res_empty = self.recommender.recommend_for_query("", top_k=5)
        self.assertEqual(len(res_empty), 0)

        # Whitespace query
        res_spaces = self.recommender.recommend_for_query("   ", top_k=5)
        self.assertEqual(len(res_spaces), 0)

        # Non-existent paper ID
        with self.assertRaises(KeyError):
            self.recommender.recommend_for_paper("non_existent_paper_id_99999")

    def test_ranking_stability(self):
        query = "reinforcement learning policy gradient actor-critic"
        run1 = self.recommender.recommend_for_query(query, top_k=5)
        run2 = self.recommender.recommend_for_query(query, top_k=5)

        ids1 = [r["paper_id"] for r in run1]
        ids2 = [r["paper_id"] for r in run2]
        self.assertEqual(ids1, ids2, "Recommendations must be completely deterministic and stable.")

    def test_compare_lexical_vs_hybrid(self):
        query = "transformers self-attention language modeling"
        comparison = self.recommender.compare_lexical_vs_hybrid(query, top_k=5)
        self.assertIn("lexical_top_k", comparison)
        self.assertIn("hybrid_top_k", comparison)
        self.assertIn("common_count", comparison)
        self.assertEqual(len(comparison["lexical_top_k"]), 5)
        self.assertEqual(len(comparison["hybrid_top_k"]), 5)


if __name__ == "__main__":
    unittest.main()

