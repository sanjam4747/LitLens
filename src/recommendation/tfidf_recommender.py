"""
TF-IDF recommendation and retrieval engine for LitLens.
Utilizes sparse matrix representations, unigram+bigram tokenization,
efficient single-query dot product cosine similarity calculations,
duplicate-version exclusion, metadata filtering, and term-level explanations.
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


class RecommendationResult(dict):
    """
    Dictionary representing a recommendation result with backward-compatible
    tuple unpacking and positional indexing support: result[0] == paper_id, result[1] == score.
    """

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            if key == 0:
                return self.get("paper_id")
            elif key == 1:
                return self.get("similarity_score", self.get("final_score", 0.0))
            raise IndexError("Index out of range for RecommendationResult positional access.")
        return super().__getitem__(key)

    def __iter__(self):
        yield self.get("paper_id")
        yield self.get("similarity_score", self.get("final_score", 0.0))


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
        self.feature_names: Optional[np.ndarray] = None

        # Optional metadata arrays for filtering
        self.paper_base_ids: Optional[np.ndarray] = None
        self.paper_categories: Optional[np.ndarray] = None
        self.paper_years: Optional[np.ndarray] = None

        self.is_fitted: bool = False

    def attach_metadata(
        self,
        base_ids: Optional[pd.Series] = None,
        categories: Optional[pd.Series] = None,
        years: Optional[pd.Series] = None,
    ) -> None:
        """
        Attaches metadata series for filtering and base_id deduplication.
        Indices must align with the index order.
        """
        if base_ids is not None:
            self.paper_base_ids = np.array(base_ids, dtype=str)
        if categories is not None:
            self.paper_categories = np.array(categories, dtype=str)
        if years is not None:
            self.paper_years = np.array(years, dtype=int)

    def fit_transform(
        self,
        corpus: pd.Series,
        paper_ids: pd.Series,
        base_ids: Optional[pd.Series] = None,
        categories: Optional[pd.Series] = None,
        years: Optional[pd.Series] = None,
    ) -> sp.csr_matrix:
        """
        Fits the TF-IDF vectorizer on the weighted corpus and transforms into a CSR sparse matrix.
        Records ID-to-index, index-to-ID mappings, and optional metadata arrays.
        """
        if len(corpus) != len(paper_ids):
            raise ValueError("Corpus and paper_ids must have identical lengths.")

        # Fit and transform
        matrix = self.vectorizer.fit_transform(corpus)
        if not sp.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()

        self.tfidf_matrix = matrix
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Build index maps
        ids_list = paper_ids.tolist()
        self.id_to_idx = {str(pid): i for i, pid in enumerate(ids_list)}
        self.idx_to_id = {i: str(pid) for i, pid in enumerate(ids_list)}

        self.attach_metadata(base_ids=base_ids, categories=categories, years=years)
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
            "paper_base_ids": self.paper_base_ids,
            "paper_categories": self.paper_categories,
            "paper_years": self.paper_years,
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
        self.paper_base_ids = metadata.get("paper_base_ids", None)
        self.paper_categories = metadata.get("paper_categories", None)
        self.paper_years = metadata.get("paper_years", None)

        if len(self.id_to_idx) != self.tfidf_matrix.shape[0]:
            raise ValueError(
                f"Mismatch: {len(self.id_to_idx)} IDs vs {self.tfidf_matrix.shape[0]} matrix rows."
            )

        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        self.is_fitted = True

    def get_overlapping_terms(
        self,
        query_vec: sp.csr_matrix,
        target_idx: int,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Computes term-level overlap contribution between a sparse query vector and a target paper.
        Contribution c_j = q_j * d_{ij}.
        Returns list of (term, contribution_weight).
        """
        if self.feature_names is None or self.tfidf_matrix is None:
            return []

        target_vec = self.tfidf_matrix[target_idx]
        # Element-wise product of query and target row
        prod = query_vec.multiply(target_vec)
        if prod.nnz == 0:
            return []

        coo = prod.tocoo()
        sort_order = np.argsort(-coo.data)[:top_n]
        results = []
        for idx in sort_order:
            term_idx = coo.col[idx]
            weight = float(coo.data[idx])
            term = self.feature_names[term_idx]
            results.append((term, round(weight, 4)))

        return results

    def _apply_filters(
        self,
        scores: np.ndarray,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
    ) -> None:
        """Applies in-place score masking (-1.0) for papers that fail category or year criteria."""
        if category and self.paper_categories is not None:
            mask = self.paper_categories != category
            scores[mask] = -1.0

        if year_min is not None and self.paper_years is not None:
            mask = self.paper_years < year_min
            scores[mask] = -1.0

        if year_max is not None and self.paper_years is not None:
            mask = self.paper_years > year_max
            scores[mask] = -1.0

    def query_similar_papers(
        self,
        paper_id: str,
        top_k: int = 10,
        exclude_self: bool = True,
        exclude_versions: bool = True,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        include_explanations: bool = False,
    ) -> List[RecommendationResult]:
        """
        Computes cosine similarity between a target paper and all catalog papers.
        Excludes the source paper and all its versions sharing base_id.
        Supports category and publication year filtering.
        Never describes scores as probabilities or accuracies.
        """
        if not self.is_fitted or self.tfidf_matrix is None:
            raise RuntimeError("Recommender is not fitted or loaded.")

        if paper_id not in self.id_to_idx:
            raise KeyError(f"Paper ID '{paper_id}' not found in index.")

        query_idx = self.id_to_idx[paper_id]
        query_vec = self.tfidf_matrix[query_idx]

        # Single-query sparse dot product
        scores = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # Always exclude exact query document if exclude_self is True
        if exclude_self:
            scores[query_idx] = -1.0

        # Exclude all papers sharing base_id if requested
        if exclude_versions and self.paper_base_ids is not None:
            source_base_id = self.paper_base_ids[query_idx]
            version_mask = self.paper_base_ids == source_base_id
            scores[version_mask] = -1.0

        # Apply category and publication year filters
        self._apply_filters(scores, category=category, year_min=year_min, year_max=year_max)

        # Select top candidates
        valid_count = int((scores > 0.0).sum())
        if valid_count == 0:
            return []

        k = min(top_k, valid_count)
        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[RecommendationResult] = []
        for idx in sorted_top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            target_pid = self.idx_to_id[idx]
            item_dict: Dict[str, Any] = {
                "paper_id": target_pid,
                "similarity_score": round(score, 4),
            }

            if self.paper_categories is not None:
                item_dict["category"] = self.paper_categories[idx]
            if self.paper_years is not None:
                item_dict["published_year"] = int(self.paper_years[idx])

            if include_explanations:
                overlapping = self.get_overlapping_terms(query_vec, idx, top_n=5)
                item_dict["overlapping_terms"] = overlapping
                terms_str = ", ".join([f"'{t}'" for t, _ in overlapping[:3]])
                item_dict["explanation"] = (
                    f"Lexical similarity score {score:.4f}. "
                    f"Strong lexical overlap in key technical concepts: {terms_str}."
                )

            results.append(RecommendationResult(item_dict))

        return results

    def query_by_text(
        self,
        query_text: str,
        top_k: int = 10,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        include_explanations: bool = False,
    ) -> List[RecommendationResult]:
        """
        Ranks papers by cosine similarity to an arbitrary natural-language query string.
        Supports category and year filtering and term-level explanations.
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

        # Apply category and publication year filters
        self._apply_filters(scores, category=category, year_min=year_min, year_max=year_max)

        valid_count = int((scores > 0.0).sum())
        if valid_count == 0:
            return []

        k = min(top_k, valid_count)
        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: List[RecommendationResult] = []
        for idx in sorted_top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue

            target_pid = self.idx_to_id[idx]
            item_dict: Dict[str, Any] = {
                "paper_id": target_pid,
                "similarity_score": round(score, 4),
            }

            if self.paper_categories is not None:
                item_dict["category"] = self.paper_categories[idx]
            if self.paper_years is not None:
                item_dict["published_year"] = int(self.paper_years[idx])

            if include_explanations:
                overlapping = self.get_overlapping_terms(query_vec, idx, top_n=5)
                item_dict["overlapping_terms"] = overlapping
                terms_str = ", ".join([f"'{t}'" for t, _ in overlapping[:3]])
                item_dict["explanation"] = (
                    f"Lexical query match {score:.4f}. "
                    f"Matched core research terms: {terms_str}."
                )

            results.append(RecommendationResult(item_dict))

        return results
