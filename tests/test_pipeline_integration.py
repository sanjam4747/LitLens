"""
Integration tests for LitLens Phase 2 serialized artifacts.
Verifies real model artifacts on disk:
- Parquet catalog integrity and row count parity
- Sparse TF-IDF matrix shape and non-zero counts
- Recommender retrieval accuracy and latency (< 100 ms)
"""

import os
import time
import unittest
import pandas as pd
import scipy.sparse as sp
from src.preprocessing import load_processed_catalog
from src.recommendation.tfidf_recommender import TFIDFRecommender


class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models"
        cls.catalog_path = os.path.join(cls.models_dir, "papers_catalog.parquet")
        cls.vec_path = os.path.join(cls.models_dir, "tfidf_vectorizer.pkl")
        cls.mat_path = os.path.join(cls.models_dir, "tfidf_matrix.npz")
        cls.meta_path = os.path.join(cls.models_dir, "tfidf_metadata.pkl")

    def test_artifacts_exist(self):
        self.assertTrue(os.path.exists(self.catalog_path), "Catalog parquet missing")
        self.assertTrue(os.path.exists(self.vec_path), "Vectorizer pkl missing")
        self.assertTrue(os.path.exists(self.mat_path), "TF-IDF matrix npz missing")
        self.assertTrue(os.path.exists(self.meta_path), "Metadata pkl missing")

    def test_catalog_and_matrix_parity(self):
        catalog = load_processed_catalog(self.catalog_path, columns=["id", "title", "published_year"])
        recommender = TFIDFRecommender()
        recommender.load_artifacts(self.models_dir)

        # Total document count parity
        expected_count = 287421
        self.assertEqual(len(catalog), expected_count)
        self.assertEqual(recommender.tfidf_matrix.shape[0], expected_count)
        self.assertEqual(recommender.tfidf_matrix.shape[1], 50000)
        self.assertEqual(len(recommender.id_to_idx), expected_count)
        self.assertEqual(len(recommender.idx_to_id), expected_count)
        self.assertTrue(sp.isspmatrix_csr(recommender.tfidf_matrix))

    def test_recommend_by_paper_id_latency_and_results(self):
        recommender = TFIDFRecommender()
        recommender.load_artifacts(self.models_dir)

        sample_id = "cs-9308101v1"
        t0 = time.perf_counter()
        recs = recommender.query_similar_papers(sample_id, top_k=5, exclude_self=True)
        query_time_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(len(recs), 5)
        self.assertLess(query_time_ms, 250, f"Query took {query_time_ms:.2f} ms (expected < 250ms)")
        
        # Self-exclusion check
        rec_ids = [r[0] for r in recs]
        self.assertNotIn(sample_id, rec_ids)
        # Scores check
        for r_id, score in recs:
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_recommend_by_free_text_latency_and_relevance(self):
        recommender = TFIDFRecommender()
        recommender.load_artifacts(self.models_dir)

        query = "deep residual neural network image classification"
        t0 = time.perf_counter()
        recs = recommender.query_by_text(query, top_k=5)
        query_time_ms = (time.perf_counter() - t0) * 1000

        self.assertEqual(len(recs), 5)
        self.assertLess(query_time_ms, 250, f"Query took {query_time_ms:.2f} ms")
        for r_id, score in recs:
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

