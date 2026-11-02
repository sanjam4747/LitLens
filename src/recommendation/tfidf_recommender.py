"""
TF-IDF recommendation and retrieval engine for LitLens.
Utilizes sparse matrix representations, unigram+bigram tokenization,
and efficient single-query dot product cosine similarity calculations.
Avoids dense pairwise similarity matrix construction.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import TOKEN_PATTERN, normalize_text


class TFIDFRecommender:
    """
    Content-based recommendation engine powered by TF-IDF and Cosine Similarity.
    Operates strictly on sparse representations (CSR format).
    """

    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.8,
        sublinear_tf: bool = True,
        stop_words: Optional[str] = "english",
        norm: str = "l2",
    ):
        self.config = {
            "max_features": max_features,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": sublinear_tf,
            "stop_words": stop_words,
            "norm": norm,
            "token_pattern": TOKEN_PATTERN,
        }

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words,
            norm=norm,
            token_pattern=TOKEN_PATTERN,
            dtype=np.float32,
        )

        self.tfidf_matrix: Optional[sp.csr_matrix] = None
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.is_fitted: bool = False

    def fit_transform(
        self,
        corpus: pd.Series,
        paper_ids: pd.Series,
    ) -> sp.csr_matrix:
        """
        Fits the TF-IDF vectorizer on the weighted corpus and transforms into a CSR sparse matrix.
        Records ID-to-index and index-to-ID mappings.
        """
        if len(corpus) != len(paper_ids):
            raise ValueError("Corpus and paper_ids must have identical lengths.")

        # Fit and transform
        matrix = self.vectorizer.fit_transform(corpus)
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        self.tfidf_matrix = matrix

        # Build index maps
        ids_list = paper_ids.tolist()
        self.id_to_idx = {str(pid): i for i, pid in enumerate(ids_list)}
        self.idx_to_id = {i: str(pid) for i, pid in enumerate(ids_list)}
        self.is_fitted = True

        return self.tfidf_matrix

    def save_artifacts(self, model_dir: str = "models") -> Dict[str, str]:
        """
        Serializes the fitted vectorizer, sparse TF-IDF matrix, and index mappings to disk.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Cannot save artifacts before fitting or loading the recommender.")

        os.makedirs(model_dir, exist_ok=True)
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(model_dir, "tfidf_matrix.npz")
        meta_path = os.path.join(model_dir, "tfidf_metadata.pkl")

        # Save vectorizer
        with open(vec_path, "wb") as f:
            pickle.dump(self.vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save sparse matrix (compressed npz)
        sp.save_npz(mat_path, self.tfidf_matrix, compressed=True)

        # Save metadata and index maps
        metadata = {
            "config": self.config,
            "matrix_shape": self.tfidf_matrix.shape,
            "nnz": int(self.tfidf_matrix.nnz),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "id_to_idx": self.id_to_idx,
            "idx_to_id": self.idx_to_id,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        return {
            "vectorizer": vec_path,
            "matrix": mat_path,
            "metadata": meta_path,
        }

    def load_artifacts(self, model_dir: str = "models") -> None:
        """
        Loads the serialized model artifacts from disk and verifies integrity.
        """
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(model_dir, "tfidf_matrix.npz")
        meta_path = os.path.join(model_dir, "tfidf_metadata.pkl")

        for p in [vec_path, mat_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required artifact at: {p}")

        with open(vec_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.tfidf_matrix = sp.load_npz(mat_path)
        if not sp.isspmatrix_csr(self.tfidf_matrix):
            self.tfidf_matrix = self.tfidf_matrix.tocsr()

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.config = metadata.get("config", self.config)
        self.id_to_idx = metadata["id_to_idx"]
        self.idx_to_id = metadata["idx_to_id"]

        if len(self.id_to_idx) != self.tfidf_matrix.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.id_to_idx)} IDs vs {self.tfidf_matrix.shape[0]} matrix rows."
            )

        self.is_fitted = True

    def query_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10,
        exclude_self: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Computes cosine similarity between a target paper and all catalog papers.
        Uses single-query sparse dot product (O(M * nnz)) without dense matrix allocation.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Recommender is not fitted or loaded.")

        if paper_id not in self.id_to_idx:
            raise KeyError(f"Paper ID '{paper_id}' not found in index.")

        query_idx = self.id_to_idx[paper_id]
        query_vec = self.tfidf_matrix[query_idx]  # 1 x V sparse CSR vector

        # L2-normalized sparse dot product equals cosine similarity
        scores = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        if exclude_self:
            scores[query_idx] = -1.0

        num_candidates = min(top_k + (1 if exclude_self else 0), len(scores))
        top_indices = np.argpartition(scores, -num_candidates)[-num_candidates:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Tuple[str, float]] = []
        for idx in sorted_top_indices:
            if exclude_self and idx == query_idx:
                continue
            if len(results) >= top_k:
                break
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.idx_to_id[idx], round(score, 4)))

        return results

    def query_by_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Ranks papers by cosine similarity to an arbitrary user query string or abstract excerpt.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Recommender is not fitted or loaded.")

        clean_text = normalize_text(query_text)
        if not clean_text:
            return []

        query_vec = self.vectorizer.transform([clean_text])
        if query_vec.nnz == 0:
            return []

        scores = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        num_candidates = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -num_candidates)[-num_candidates:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[Tuple[str, float]] = []
        for idx in sorted_top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.idx_to_id[idx], round(score, 4)))

        return results

