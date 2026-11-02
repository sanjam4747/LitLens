"""
Script to train the TopicModeler (LDA) using existing Phase 2 TF-IDF matrix and vectorizer.
Transforms the entire 287,421 paper corpus in batches and serializes:
- models/lda_model.pkl
- models/topic_distributions.npy
- models/topic_metadata.pkl
"""

import os
import sys
import time
import pickle
import scipy.sparse as sp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.recommendation.topic_modeler import TopicModeler
from src.utils.metrics import log_stage, get_memory_usage_mb


def main():
    print("=" * 70)
    print("  LitLens Phase 3: LDA Topic Model Training & Corpus Transformation")
    print("=" * 70)

    models_dir = os.path.join(PROJECT_ROOT, "models")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    mat_path = os.path.join(models_dir, "tfidf_matrix.npz")
    meta_path = os.path.join(models_dir, "tfidf_metadata.pkl")

    with log_stage("Loading Phase 2 TF-IDF Artifacts"):
        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        feature_names = vectorizer.get_feature_names_out()

        tfidf_matrix = sp.load_npz(mat_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        paper_ids = list(meta["id_to_idx"].keys())
        print(f"Loaded matrix: {tfidf_matrix.shape} with {len(paper_ids):,} paper IDs.")

    with log_stage("Training LDA Topic Model (16 Topics, Online Variational Inference)"):
        tm = TopicModeler(
            n_components=16,
            max_iter=10,
            batch_size=2048,
            random_state=42,
            top_features=5000,
            max_doc_ratio=0.25,
            min_doc_count=10,
        )
        tm.fit(tfidf_matrix, feature_names, sample_size=40000)

        print("\nDiscovered Topics Summary:")
        for topic_id, t_info in tm.get_all_topics().items():
            top_words = ", ".join([f"{term}" for term, _ in t_info["top_terms"][:6]])
            print(f"  Topic {topic_id+1:02d} [{t_info['label']}]: {top_words}")

    with log_stage("Batch Transforming Corpus into Normalized Topic Distributions"):
        distributions = tm.transform_corpus(tfidf_matrix, paper_ids, batch_size=15000)
        print(f"Generated normalized topic distributions: {distributions.shape} ({distributions.nbytes / 1024**2:.2f} MB)")

    with log_stage("Serializing Topic Model Artifacts"):
        saved = tm.save_artifacts(models_dir)
        for name, p in saved.items():
            sz_mb = os.path.getsize(p) / 1024**2
            print(f"  Saved '{name}': {p} ({sz_mb:.2f} MB)")

    # Verify reload
    with log_stage("Verifying Topic Artifacts Reload"):
        verifier = TopicModeler()
        verifier.load_artifacts(models_dir)
        assert verifier.topic_distributions.shape == (len(paper_ids), 16)
        print(f"Verification successful: loaded {len(verifier.id_to_idx):,} document distributions.")

    print("\n" + "=" * 70)
    print("  LDA Topic Modeling & Indexing Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

