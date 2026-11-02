"""
LitLens Phase 2 End-to-End Pipeline Execution Script.
Executes:
1. Dataset ingestion & schema validation
2. Comprehensive dataset quality inspection & report generation
3. Metadata standardization & Parquet catalog export
4. Weighted soup feature engineering
5. TF-IDF vectorization & sparse CSR matrix construction
6. Artifact serialization with execution time and memory profiling
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_raw_dataset,
    inspect_dataset_quality,
    generate_markdown_quality_report,
)
from src.preprocessing import (
    standardize_metadata,
    build_weighted_soup,
    save_processed_catalog,
)
from src.recommendation.tfidf_recommender import TFIDFRecommender
from src.utils.metrics import log_stage, get_memory_usage_mb


def main():
    print("=" * 70)
    print("  LitLens Phase 2: Dataset Processing & Core NLP Indexing Pipeline")
    print("=" * 70)

    dataset_path = os.path.join(PROJECT_ROOT, "data", "arXiv_scientific_dataset.csv")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    pipeline_stats = {}
    t_pipeline_start = time.perf_counter()

    # --- STAGE 1: Dataset Loading & Quality Validation ---
    with log_stage("Stage 1: Dataset Ingestion & Quality Validation") as stage_info:
        df_raw = load_raw_dataset(dataset_path)
        print(f"Loaded {len(df_raw):,} records across {len(df_raw.columns)} columns.")

        quality_report = inspect_dataset_quality(df_raw)
        report_md = generate_markdown_quality_report(quality_report)
        report_path = os.path.join(reports_dir, "dataset_quality_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Generated dataset quality report at: {report_path}")
        pipeline_stats["stage_1"] = stage_info

    # --- STAGE 2: Preprocessing & Catalog Standardization ---
    with log_stage("Stage 2: Metadata Standardization & Parquet Export") as stage_info:
        df_clean = standardize_metadata(df_raw)
        catalog_path = os.path.join(models_dir, "papers_catalog.parquet")
        save_processed_catalog(df_clean, catalog_path)
        catalog_size_mb = os.path.getsize(catalog_path) / (1024 * 1024)
        print(f"Saved processed catalog to {catalog_path} ({catalog_size_mb:.2f} MB)")
        pipeline_stats["stage_2"] = stage_info

    # Release df_raw memory early
    del df_raw

    # --- STAGE 3: Weighted Soup Feature Engineering ---
    with log_stage("Stage 3: Weighted Text Soup Construction") as stage_info:
        # Title weight = 3, includes category and summary abstract
        soup = build_weighted_soup(df_clean, title_weight=3, include_category=True)
        paper_ids = df_clean["id"]
        print(f"Constructed weighted soup for {len(soup):,} documents.")
        pipeline_stats["stage_3"] = stage_info

    # --- STAGE 4: TF-IDF Sparse Vectorization ---
    with log_stage("Stage 4: TF-IDF Vectorization & Sparse Indexing") as stage_info:
        recommender = TFIDFRecommender(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            sublinear_tf=True,
            stop_words="english",
            norm="l2",
        )
        matrix = recommender.fit_transform(soup, paper_ids)
        print(f"TF-IDF sparse matrix shape: {matrix.shape}")
        print(f"Non-zero elements: {matrix.nnz:,} (Density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.6f})")
        pipeline_stats["stage_4"] = stage_info

    # --- STAGE 5: Artifact Serialization & Verification ---
    with log_stage("Stage 5: Artifact Serialization & Verification") as stage_info:
        saved_paths = recommender.save_artifacts(models_dir)
        for name, p in saved_paths.items():
            sz_mb = os.path.getsize(p) / (1024 * 1024)
            print(f"Saved artifact '{name}': {p} ({sz_mb:.2f} MB)")

        # Verify reload
        verifier = TFIDFRecommender()
        verifier.load_artifacts(models_dir)
        print(f"Verification successful: Loaded index with {len(verifier.id_to_idx):,} documents.")
        assert verifier.tfidf_matrix.shape == matrix.shape
        assert len(verifier.id_to_idx) == len(df_clean)
        pipeline_stats["stage_5"] = stage_info

    # Summary
    total_time = time.perf_counter() - t_pipeline_start
    final_mem = get_memory_usage_mb()
    print("\n" + "=" * 70)
    print(f"  Pipeline Finished Successfully in {total_time:.2f}s (~{total_time/60:.2f} min)")
    print(f"  Final Process Memory: {final_mem:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()

