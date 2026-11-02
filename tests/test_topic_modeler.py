"""
Unit tests for TopicModeler (LDA) engine.
Tests model persistence, topic discovery, distribution normalization,
and query projection.
"""

import os
import unittest
import numpy as np
import scipy.sparse as sp
from src.recommendation.topic_modeler import TopicModeler


class TestTopicModeler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = "models"
        cls.modeler = TopicModeler()
        cls.modeler.load_artifacts(cls.models_dir)

    def test_artifacts_loaded(self):
        self.assertTrue(self.modeler.is_fitted)
        self.assertIsNotNone(self.modeler.lda)
        self.assertIsNotNone(self.modeler.topic_distributions)
        self.assertEqual(self.modeler.n_components, 16)
        self.assertEqual(self.modeler.topic_distributions.shape[0], 287421)
        self.assertEqual(self.modeler.topic_distributions.shape[1], 16)

    def test_distribution_l2_normalization(self):
        # Sample 500 rows and verify L2 norms are 1.0 (within float32 precision)
        sample = self.modeler.topic_distributions[:500]
        norms = np.linalg.norm(sample, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_get_all_topics(self):
        topics = self.modeler.get_all_topics()
        self.assertEqual(len(topics), 16)
        for tid, t_info in topics.items():
            self.assertIn("label", t_info)
            self.assertIn("top_terms", t_info)
            self.assertGreater(len(t_info["top_terms"]), 0)

    def test_get_topic_terms(self):
        terms = self.modeler.get_topic_terms(0, top_n=5)
        self.assertEqual(len(terms), 5)
        for term, weight in terms:
            self.assertIsInstance(term, str)
            self.assertGreater(weight, 0.0)

    def test_get_paper_dominant_topic(self):
        sample_id = "cs-9308101v1"
        dom = self.modeler.get_paper_dominant_topic(sample_id)
        self.assertIn("topic_id", dom)
        self.assertIn("topic_weight", dom)
        self.assertIn("label", dom)
        self.assertIn("top_terms", dom)
        self.assertGreaterEqual(dom["topic_id"], 0)
        self.assertLess(dom["topic_id"], 16)
        self.assertGreater(dom["topic_weight"], 0.0)

    def test_discover_papers_by_topic(self):
        discovered = self.modeler.discover_papers_by_topic(topic_id=0, top_k=5)
        self.assertEqual(len(discovered), 5)
        for pid, weight in discovered:
            self.assertIn(pid, self.modeler.id_to_idx)
            self.assertGreater(weight, 0.0)
            self.assertLessEqual(weight, 1.0)

    def test_compute_topic_similarities_bounds(self):
        # Query with random normalized vector
        q_vec = np.zeros((1, 16), dtype=np.float32)
        q_vec[0, 2] = 1.0
        scores = self.modeler.compute_topic_similarities(q_vec)
        self.assertEqual(len(scores), 287421)
        self.assertGreaterEqual(scores.min(), 0.0)
        self.assertLessEqual(scores.max(), 1.0)


if __name__ == "__main__":
    unittest.main()

