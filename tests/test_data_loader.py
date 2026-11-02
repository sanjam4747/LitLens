"""
Unit tests for data_loader module.
Tests dataset schema validation, missing value detection, and base_id extraction.
"""

import os
import unittest
import pandas as pd
from src.data_loader import (
    validate_schema,
    extract_base_id_and_version,
    inspect_dataset_quality,
    load_raw_dataset,
    EXPECTED_COLUMNS,
)


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a sample synthetic DataFrame mimicking arXiv dataset schema
        self.sample_df = pd.DataFrame({
            "id": ["cs-9308101v1", "abs-2501.14993v1", "abs-2501.14993v2", "abs-1706.03762v1"],
            "title": [
                "Dynamic Backtracking",
                "Attention Is All You Need",
                "Attention Is All You Need Revised",
                "Deep Residual Learning",
            ],
            "category": [
                "Artificial Intelligence",
                "Computation and Language",
                "Computation and Language",
                "Computer Vision",
            ],
            "category_code": ["cs.AI", "cs.CL", "cs.CL", "cs.CV"],
            "published_date": ["8/1/93", "1/25/25", "1/25/25", "6/18/17"],
            "updated_date": ["8/1/93", "1/25/25", "02/13/25", "6/18/17"],
            "authors": [
                "['M. L. Ginsberg']",
                "['A. Vaswani', 'N. Shazeer']",
                "['A. Vaswani', 'N. Shazeer']",
                "['K. He', 'X. Zhang']",
            ],
            "first_author": ["M. L. Ginsberg", "A. Vaswani", "A. Vaswani", "K. He"],
            "summary": [
                "Because of their occasional need to return to shallow points in search.",
                "The dominant sequence transduction models are based on recurrent networks.",
                "The dominant sequence transduction models are based on complex recurrent networks.",
                "Deeper neural networks are more difficult to train effectively.",
            ],
            "summary_word_count": [11, 10, 11, 9],
        })

    def test_schema_validation_valid(self):
        is_valid, missing = validate_schema(self.sample_df)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

    def test_schema_validation_missing_col(self):
        incomplete_df = self.sample_df.drop(columns=["summary"])
        is_valid, missing = validate_schema(incomplete_df)
        self.assertFalse(is_valid)
        self.assertIn("summary", missing)

    def test_extract_base_id_and_version(self):
        base_id, ver = extract_base_id_and_version("abs-2501.14993v3")
        self.assertEqual(base_id, "abs-2501.14993")
        self.assertEqual(ver, 3)

        base_id, ver = extract_base_id_and_version("cs-9308101v1")
        self.assertEqual(base_id, "cs-9308101")
        self.assertEqual(ver, 1)

        base_id, ver = extract_base_id_and_version("1706.03762")
        self.assertEqual(base_id, "1706.03762")
        self.assertEqual(ver, 1)

    def test_inspect_dataset_quality(self):
        report = inspect_dataset_quality(self.sample_df)
        self.assertEqual(report["total_records"], 4)
        self.assertTrue(report["is_valid_schema"])
        self.assertEqual(report["id_stats"]["unique_ids"], 4)
        self.assertEqual(report["id_stats"]["duplicate_ids"], 0)
        self.assertEqual(report["id_stats"]["unique_base_ids"], 3)
        self.assertEqual(report["id_stats"]["duplicate_base_id_rows"], 1)
        self.assertEqual(report["summary_stats"]["empty_summaries"], 0)
        self.assertEqual(report["date_stats"]["failed_pub_dates"], 0)

    def test_load_raw_dataset_sample(self):
        # Test loading top 5 rows from actual dataset file
        csv_path = "data/arXiv_scientific_dataset.csv"
        if os.path.exists(csv_path):
            df = load_raw_dataset(csv_path, nrows=5)
            self.assertEqual(len(df), 5)
            for col in EXPECTED_COLUMNS:
                self.assertIn(col, df.columns)


if __name__ == "__main__":
    unittest.main()

