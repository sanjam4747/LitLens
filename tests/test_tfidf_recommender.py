"""
Unit tests for TFIDFRecommender.
Tests vectorizer fitting, sparse CSR matrix construction, artifact persistence,
paper-to-paper query retrieval, and arbitrary text query matching.
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from src.recommendation.tfidf_recommender import TFIDFRecommender


class TestTFIDFRecommender(unittest.TestCase):

    def setUp(self):
        self.corpus = pd.Series([
            "attention transformer self-attention neural network language model computation",
            "transformer attention mechanism bert deep neural language understanding",
            "convolutional neural network image classification computer vision resnet",
            "vision transformer deep learning convolutional network object detection",
            "quantum computing quantum circuit entanglement physics simulation",
        ])
        self.paper_ids = pd.Series(["p1", "p2", "p3", "p4", "p5"])

    def test_fit_transform(self):
        rec = TFIDFRecommender(max_features=100, ngram_range=(1, 2), min_df=1)
        matrix = rec.fit_transform(self.corpus, self.paper_ids)

        self.assertTrue(rec.is_fitted)
        self.assertTrue(sp.isspmatrix_csr(matrix))
        self.assertEqual(matrix.shape[0], 5)
        self.assertEqual(len(rec.id_to_idx), 5)
        self.assertEqual(len(rec.idx_to_id), 5)

        # Check bigrams exist in vocabulary
        vocab = rec.vectorizer.vocabulary_
        has_bigram = any(" " in term for term in vocab.keys())
        self.assertTrue(has_bigram, "Vocabulary should contain bigrams when ngram_range=(1, 2)")

    def test_save_and_load_artifacts(self):
        rec = TFIDFRecommender(max_features=100, ngram_range=(1, 2), min_df=1)
        matrix = rec.fit_transform(self.corpus, self.paper_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            rec.save_artifacts(tmpdir)

            # Verify files on disk
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tfidf_vectorizer.pkl")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tfidf_matrix.npz")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "tfidf_metadata.pkl")))

            # Load into new instance
            rec2 = TFIDFRecommender()
            rec2.load_artifacts(tmpdir)

            self.assertTrue(rec2.is_fitted)
            self.assertEqual(rec2.tfidf_matrix.shape, matrix.shape)
            self.assertEqual(rec2.tfidf_matrix.nnz, matrix.nnz)
            self.assertEqual(rec2.id_to_idx, rec.id_to_idx)
            self.assertEqual(rec2.idx_to_id, rec.idx_to_id)

    def test_query_similar_papers(self):
        rec = TFIDFRecommender(max_features=100, ngram_range=(1, 2), min_df=1)
        rec.fit_transform(self.corpus, self.paper_ids)

        # p1 (transformer language) should be most similar to p2 (transformer language)
        recs = rec.query_similar_papers("p1", top_k=2, exclude_self=True)
        self.assertGreater(len(recs), 0)
        top_match_id, top_score = recs[0]
        self.assertEqual(top_match_id, "p2")
        self.assertGreater(top_score, 0.0)
        self.assertLessEqual(top_score, 1.0)

        # Verify self is excluded
        returned_ids = [r[0] for r in recs]
        self.assertNotIn("p1", returned_ids)

    def test_query_by_text(self):
        rec = TFIDFRecommender(max_features=100, ngram_range=(1, 2), min_df=1)
        rec.fit_transform(self.corpus, self.paper_ids)

        recs = rec.query_by_text("quantum circuit entanglement", top_k=2)
        self.assertGreater(len(recs), 0)
        self.assertEqual(recs[0][0], "p5")
        self.assertGreater(recs[0][1], 0.0)

    def test_empty_or_unseen_query(self):
        rec = TFIDFRecommender(max_features=100, ngram_range=(1, 2), min_df=1)
        rec.fit_transform(self.corpus, self.paper_ids)

        # Completely out of vocabulary query
        recs = rec.query_by_text("zyxw vutsrqpon", top_k=2)
        self.assertEqual(len(recs), 0)

        # Empty query
        recs_empty = rec.query_by_text("", top_k=2)
        self.assertEqual(len(recs_empty), 0)


if __name__ == "__main__":
    unittest.main()

