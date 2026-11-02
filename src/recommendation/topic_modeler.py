"""
Topic modeling engine for LitLens using Latent Dirichlet Allocation (LDA).
Implements resource-conscious online variational inference, batch transformation,
sparse representations, topic persistence, and topic-based paper discovery.
"""

import os
import pickle
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import LatentDirichletAllocation

from src.preprocessing import normalize_text


class TopicModeler:
    """
    Latent Dirichlet Allocation (LDA) Topic Modeler for scientific paper discovery.
    Trained on informative feature slices and persisted for instant topic similarity retrieval.
    """

    def __init__(
        self,
        n_components: int = 16,
        max_iter: int = 10,
        batch_size: int = 2048,
        random_state: int = 42,
        top_features: int = 5000,
        max_doc_ratio: float = 0.25,
        min_doc_count: int = 10,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.top_features = top_features
        self.max_doc_ratio = max_doc_ratio
        self.min_doc_count = min_doc_count

        self.lda: Optional[LatentDirichletAllocation] = None
        self.feature_indices: Optional[np.ndarray] = None
        self.feature_names: Optional[np.ndarray] = None
        self.topic_distributions: Optional[np.ndarray] = None  # Normalized N x K array
        self.topic_summaries: Dict[int, Dict[str, Any]] = {}

        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.is_fitted: bool = False

    def _select_informative_features(
        self,
        sparse_matrix: sp.csr_matrix,
        all_feature_names: np.ndarray,
    ) -> np.ndarray:
        """
        Prunes corpus-ubiquitous terms (e.g. appearing in > 25% docs) and rare noise (< 10 docs).
        Selects the top informative features by overall column frequency.
        """
        num_docs = sparse_matrix.shape[0]
        doc_counts = (sparse_matrix > 0).sum(axis=0).A1
        col_sums = sparse_matrix.sum(axis=0).A1

        # Mask ubiquitous and rare features
        mask = (doc_counts < self.max_doc_ratio * num_docs) & (doc_counts >= self.min_doc_count)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(sparse_matrix.shape[1])

        # Pick top informative features
        n_keep = min(self.top_features, len(valid_indices))
        selected = valid_indices[col_sums[valid_indices].argsort()[::-1][:n_keep]]
        selected.sort()
        return selected

    def fit(
        self,
        sparse_matrix: sp.csr_matrix,
        all_feature_names: np.ndarray,
        sample_size: int = 40000,
    ) -> "TopicModeler":
        """
        Fits the LDA model on a uniform representative sample of the sparse matrix.
        Uses online variational inference with batch processing.
        """
        if not sp.isspmatrix_csr(sparse_matrix):
            sparse_matrix = sparse_matrix.tocsr()

        # Feature selection
        self.feature_indices = self._select_informative_features(sparse_matrix, all_feature_names)
        self.feature_names = all_feature_names[self.feature_indices]

        # Slice sub-matrix of informative features
        X_sub = sparse_matrix[:, self.feature_indices]

        # Uniform random sampling across corpus
        num_docs = X_sub.shape[0]
        rng = np.random.default_rng(self.random_state)
        if sample_size < num_docs:
            sample_idx = rng.choice(num_docs, size=sample_size, replace=False)
            train_data = X_sub[sample_idx]
        else:
            train_data = X_sub

        self.lda = LatentDirichletAllocation(
            n_components=self.n_components,
            max_iter=self.max_iter,
            learning_method="online",
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.lda.fit(train_data)
        self._build_topic_summaries()
        self.is_fitted = True
        return self

    def _build_topic_summaries(self) -> None:
        """Constructs human-interpretable labels and top keywords for each topic."""
        if self.lda is None or self.feature_names is None:
            return

        self.topic_summaries = {}
        for k, comp in enumerate(self.lda.components_):
            top_kw_idx = comp.argsort()[:-11:-1]
            top_terms = [
                (self.feature_names[i], round(float(comp[i] / comp.sum()), 4))
                for i in top_kw_idx
            ]
            # Form clean descriptive label from top 3 words
            label_terms = [t[0] for t in top_terms[:3]]
            label = " / ".join(label_terms).title()

            self.topic_summaries[k] = {
                "topic_id": k,
                "label": label,
                "top_terms": top_terms,
            }

    def transform_corpus(
        self,
        sparse_matrix: sp.csr_matrix,
        paper_ids: List[str],
        batch_size: int = 15000,
    ) -> np.ndarray:
        """
        Transforms all documents into normalized topic distributions in batches.
        Row vectors are L2-normalized so that cosine similarity equals dot product.
        """
        if not self.is_fitted or self.lda is None or self.feature_indices is None:
            raise RuntimeError("TopicModeler must be fitted before transforming.")

        if not sp.isspmatrix_csr(sparse_matrix):
            sparse_matrix = sparse_matrix.tocsr()

        X_sub = sparse_matrix[:, self.feature_indices]
        num_docs = X_sub.shape[0]

        # Batch transform
        raw_distributions = np.empty((num_docs, self.n_components), dtype=np.float32)
        for start_idx in range(0, num_docs, batch_size):
            end_idx = min(start_idx + batch_size, num_docs)
            batch = X_sub[start_idx:end_idx]
            raw_distributions[start_idx:end_idx] = self.lda.transform(batch).astype(np.float32)

        # L2-normalize each row for fast cosine similarity: sim = dot(v1, v2)
        norms = np.linalg.norm(raw_distributions, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.topic_distributions = raw_distributions / norms

        # Build index maps
        self.id_to_idx = {str(pid): i for i, pid in enumerate(paper_ids)}
        self.idx_to_id = {i: str(pid) for i, pid in enumerate(paper_ids)}

        return self.topic_distributions

    def transform_text_query(
        self,
        query_vec_sparse: sp.csr_matrix,
    ) -> np.ndarray:
        """
        Projects a TF-IDF sparse vector into the L2-normalized topic space.
        Returns 1 x K float32 array.
        """
        if not self.is_fitted or self.lda is None or self.feature_indices is None:
            raise RuntimeError("TopicModeler is not fitted.")

        sub_vec = query_vec_sparse[:, self.feature_indices]
        if sub_vec.nnz == 0:
            # Fallback to uniform distribution if no topic terms match
            unif = np.ones((1, self.n_components), dtype=np.float32)
            return unif / np.linalg.norm(unif)

        raw_dist = self.lda.transform(sub_vec).astype(np.float32)
        norm = np.linalg.norm(raw_dist)
        if norm > 0:
            raw_dist /= norm
        return raw_dist

    def save_artifacts(self, model_dir: str = "models") -> Dict[str, str]:
        """Serializes fitted LDA model, topic distributions, and metadata."""
        if not self.is_fitted or self.lda is None or self.topic_distributions is None:
            raise RuntimeError("Cannot save artifacts before model is fitted and corpus transformed.")

        os.makedirs(model_dir, exist_ok=True)
        lda_path = os.path.join(model_dir, "lda_model.pkl")
        dist_path = os.path.join(model_dir, "topic_distributions.npy")
        meta_path = os.path.join(model_dir, "topic_metadata.pkl")

        with open(lda_path, "wb") as f:
            pickle.dump(self.lda, f, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(dist_path, self.topic_distributions)

        meta = {
            "n_components": self.n_components,
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
            "topic_summaries": self.topic_summaries,
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "lda_model": lda_path,
            "topic_distributions": dist_path,
            "topic_metadata": meta_path,
        }

    def load_artifacts(self, model_dir: str = "models") -> None:
        """Loads serialized topic model and distributions from disk."""
        lda_path = os.path.join(model_dir, "lda_model.pkl")
        dist_path = os.path.join(model_dir, "topic_distributions.npy")
        meta_path = os.path.join(model_dir, "topic_metadata.pkl")

        for p in [lda_path, dist_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required topic artifact: {p}")

        with open(lda_path, "rb") as f:
            self.lda = pickle.load(f)

        self.topic_distributions = np.load(dist_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.n_components = meta["n_components"]
        self.feature_indices = meta["feature_indices"]
        self.feature_names = meta["feature_names"]
        self.topic_summaries = meta["topic_summaries"]
        self.id_to_idx = meta["id_to_idx"]
        self.idx_to_id = meta["idx_to_id"]
        self.is_fitted = True

    def get_topic_terms(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """Returns top terms and relative weights for a specific topic."""
        if topic_id not in self.topic_summaries:
            raise KeyError(f"Topic ID {topic_id} not found.")
        return self.topic_summaries[topic_id]["top_terms"][:top_n]

    def get_all_topics(self) -> Dict[int, Dict[str, Any]]:
        """Returns metadata and top keywords for all topics."""
        return self.topic_summaries

    def get_paper_dominant_topic(self, paper_id: str) -> Dict[str, Any]:
        """Returns the dominant topic, its normalized weight, and topic label for a given paper."""
        if paper_id not in self.id_to_idx or self.topic_distributions is None:
            raise KeyError(f"Paper '{paper_id}' not found in topic index.")

        idx = self.id_to_idx[paper_id]
        dist = self.topic_distributions[idx]
        top_topic = int(np.argmax(dist))
        weight = float(dist[top_topic])
        summary = self.topic_summaries.get(top_topic, {})

        return {
            "topic_id": top_topic,
            "topic_weight": round(weight, 4),
            "label": summary.get("label", f"Topic {top_topic:02d}"),
            "top_terms": [t[0] for t in summary.get("top_terms", [])[:5]],
        }

    def discover_papers_by_topic(
        self,
        topic_id: int,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Discovers papers having the highest representation in a given topic.
        Returns list of (paper_id, topic_weight).
        """
        if self.topic_distributions is None or not self.is_fitted:
            raise RuntimeError("TopicModeler is not fitted or loaded.")

        if topic_id < 0 or topic_id >= self.n_components:
            raise ValueError(f"Topic ID must be between 0 and {self.n_components - 1}.")

        col = self.topic_distributions[:, topic_id]
        top_indices = np.argpartition(col, -top_k)[-top_k:]
        sorted_top_indices = top_indices[np.argsort(-col[top_indices])]

        results: List[Tuple[str, float]] = []
        for idx in sorted_top_indices:
            weight = float(col[idx])
            results.append((self.idx_to_id[idx], round(weight, 4)))

        return results

    def compute_topic_similarities(self, query_topic_vec: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity of topic distributions:
        sim = dot(topic_distributions, query_topic_vec.T).
        Both are L2-normalized, guaranteeing sim in [0, 1].
        """
        if self.topic_distributions is None:
            raise RuntimeError("Topic distributions are not loaded.")

        # Matrix dot product (N x K) . (K x 1) -> (N,)
        scores = self.topic_distributions.dot(query_topic_vec.T).ravel()
        # Clip numerical inaccuracies to [0.0, 1.0]
        return np.clip(scores, 0.0, 1.0)

