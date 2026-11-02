"""
Unit tests for preprocessing module.
Tests text normalization, LaTeX/HTML cleaning, metadata standardization,
weighted soup construction, and Parquet catalog serialization.
"""

import os
import tempfile
import unittest
import pandas as pd
from src.preprocessing import (
    normalize_text,
    standardize_metadata,
    build_weighted_soup,
    save_processed_catalog,
    load_processed_catalog,
)


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "id": ["cs-01v1", "cs-01v2"],
            "title": [
                "Attention Is All You Need for $O(N)$ Transformers",
                "Attention Is All You Need for $O(N)$ Transformers (v2)",
            ],
            "category": ["Computation and Language", "Computation and Language"],
            "category_code": ["cs.CL", "cs.CL"],
            "published_date": ["6/12/17", "6/12/17"],
            "updated_date": ["6/12/17", "12/06/17"],
            "authors": ["['A. Vaswani', 'N. Shazeer']", "['A. Vaswani', 'N. Shazeer']"],
            "first_author": ["A. Vaswani", "A. Vaswani"],
            "summary": [
                "We propose the \\textbf{Transformer}, a model architecture relying entirely on attention.",
                "We propose the Transformer model with updated benchmarks on WMT 2014 &amp; En-De.",
            ],
            "summary_word_count": [11, 12],
        })

    def test_normalize_text_latex_and_html(self):
        raw = "We propose \\textbf{BERT} with $\\mathcal{O}(N)$ complexity &amp; 3D-CNNs."
        norm = normalize_text(raw)
        # Should strip \textbf{}, decode &amp; to &, lowercase, preserve bert, o(n), 3d-cnns
        self.assertIn("bert", norm)
        self.assertIn("3d-cnns", norm)
        self.assertNotIn("\\textbf", norm)
        self.assertNotIn("&amp;", norm)

    def test_normalize_text_scientific_acronyms(self):
        raw = "State-of-the-art ResNet-50 architecture for deep learning."
        norm = normalize_text(raw)
        self.assertIn("state-of-the-art", norm)
        self.assertIn("resnet-50", norm)

    def test_standardize_metadata(self):
        std_df = standardize_metadata(self.sample_df)
        # Check original columns preserved
        self.assertIn("title", std_df.columns)
        self.assertIn("summary", std_df.columns)
        self.assertIn("authors", std_df.columns)
        # Check derived columns
        self.assertIn("published_year", std_df.columns)
        self.assertEqual(int(std_df.iloc[0]["published_year"]), 2017)
        self.assertIn("base_id", std_df.columns)
        self.assertEqual(std_df.iloc[0]["base_id"], "cs-01")
        self.assertEqual(std_df.iloc[0]["version"], 1)
        self.assertEqual(std_df.iloc[1]["version"], 2)
        # Check latest version flag
        self.assertFalse(std_df.iloc[0]["is_latest_version"])
        self.assertTrue(std_df.iloc[1]["is_latest_version"])
        # Check clean authors display
        self.assertEqual(std_df.iloc[0]["authors_display"], "A. Vaswani, N. Shazeer")

    def test_build_weighted_soup(self):
        soup = build_weighted_soup(self.sample_df, title_weight=3, include_category=True)
        self.assertEqual(len(soup), 2)
        # Check title word frequency in soup is greater than 1
        first_doc = soup.iloc[0]
        self.assertGreaterEqual(first_doc.count("attention"), 3)
        self.assertIn("computation and language", first_doc)
        self.assertIn("transformer", first_doc)

    def test_parquet_catalog_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_file = os.path.join(tmpdir, "test_catalog.parquet")
            save_processed_catalog(self.sample_df, catalog_file)
            self.assertTrue(os.path.exists(catalog_file))

            loaded = load_processed_catalog(catalog_file)
            self.assertEqual(len(loaded), len(self.sample_df))
            self.assertEqual(list(loaded.columns), list(self.sample_df.columns))
            self.assertEqual(loaded.iloc[0]["title"], self.sample_df.iloc[0]["title"])


if __name__ == "__main__":
    unittest.main()

