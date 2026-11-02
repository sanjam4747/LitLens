"""
Hybrid Recommendation Engine for LitLens.
Combines TF-IDF lexical cosine similarity with LDA topic-distribution similarity
using configurable score fusion, component score exposure, metadata filtering,
and explainability.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.preprocessing import load_processed_catalog, normalize_text
from src.recommendation.tfidf_recommender import TFIDFRecommender, RecommendationResult
from src.recommendation.topic_modeler import TopicModeler


class HybridRecommender:
    """
    Hybrid discovery engine combining sparse TF-IDF lexical matching
    and latent topic distribution alignment.
    """

    def __init__(
        self,
        tfidf_recommender: Optional[TFIDFRecommender] = None,
        topic_modeler: Optional[TopicModeler] = None,
        catalog: Optional[pd.DataFrame] = None,
        models_dir: str = "models",
    ):
        self.models_dir = models_dir

        if tfidf_recommender is not None:
            self.tfidf = tfidf_recommender
        else:
            self.tfidf = TFIDFRecommender()
            self.tfidf.load_artifacts(models_dir)

        if topic_modeler is not None:
            self.topic = topic_modeler
        else:
            self.topic = TopicModeler()
            self.topic.load_artifacts(models_dir)

        if catalog is not None:
            self.catalog = catalog
        else:
            catalog_path = f"{models_dir}/papers_catalog.parquet"
            self.catalog = load_processed_catalog(
                catalog_path,
                columns=[
                    "id",
                    "base_id",
                    "title",
                    "category",
                    "published_year",
                    "authors_display",
                    "summary",
                ],
            )

        # Index lookup dataframe by paper ID
        self.catalog_by_id = self.catalog.set_index("id", drop=False)

    def _get_paper_metadata(self, paper_id: str) -> Dict[str, Any]:
        """Safely fetches metadata dictionary for a paper."""
        if paper_id in self.catalog_by_id.index:
            row = self.catalog_by_id.loc[paper_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {
                "paper_id": str(paper_id),
                "title": str(row["title"]),
                "category": str(row["category"]),
                "published_year": int(row["published_year"]),
                "authors": str(row["authors_display"]),
                "base_id": str(row["base_id"]),
                "summary": str(row["summary"]),
            }
        return {
            "paper_id": str(paper_id),
            "title": "Unknown Title",
            "category": "Unknown",
            "published_year": 0,
            "authors": "Unknown",
            "base_id": str(paper_id),
            "summary": "",
        }

    def recommend_for_paper(
        self,
        paper_id: str,
        top_k: int = 10,
        weight_tfidf: float = 0.6,
        weight_topic: float = 0.4,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        candidate_pool_size: int = 400,
    ) -> List[RecommendationResult]:
        """
        Generates hybrid recommendations for a given paper.
        Fuses TF-IDF lexical similarity with topic distribution similarity.
        Excludes the source paper and all its versions sharing base_id.
        Exposes component scores independently.
        """
        if paper_id not in self.tfidf.id_to_idx:
            raise KeyError(f"Paper '{paper_id}' not found in catalog.")

        # Normalize fusion weights
        total_w = weight_tfidf + weight_topic
        if total_w <= 0:
            weight_tfidf, weight_topic = 0.5, 0.5
            total_w = 1.0
        w_tfidf = weight_tfidf / total_w
        w_topic = weight_topic / total_w

        src_meta = self._get_paper_metadata(paper_id)
        src_base_id = src_meta["base_id"]
        query_idx = self.tfidf.id_to_idx[paper_id]
        query_vec = self.tfidf.tfidf_matrix[query_idx]

        # 1. Compute TF-IDF scores
        tfidf_scores = self.tfidf.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # 2. Compute Topic similarity scores
        query_topic_vec = self.topic.topic_distributions[query_idx: query_idx + 1]
        topic_scores = self.topic.compute_topic_similarities(query_topic_vec)

        # 3. Exclude source and its versions
        if self.tfidf.paper_base_ids is not None:
            version_mask = self.tfidf.paper_base_ids == src_base_id
            tfidf_scores[version_mask] = -1.0
            topic_scores[version_mask] = -1.0
        else:
            tfidf_scores[query_idx] = -1.0
            topic_scores[query_idx] = -1.0

        # 4. Apply metadata filters
        self.tfidf._apply_filters(tfidf_scores, category=category, year_min=year_min, year_max=year_max)
        self.tfidf._apply_filters(topic_scores, category=category, year_min=year_min, year_max=year_max)

        # 5. Hybrid fused score: w_tfidf * s_lex + w_topic * s_top
        valid_mask = (tfidf_scores >= 0) & (topic_scores >= 0)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return []

        # Restrict to top candidate pool from both signals for fast scoring
        candidate_k = min(candidate_pool_size, len(valid_indices))
        pool_indices = np.argpartition(tfidf_scores, -candidate_k)[-candidate_k:]

        fused_scores = {}
        for idx in pool_indices:
            s_lex = float(tfidf_scores[idx])
            s_top = float(topic_scores[idx])
            if s_lex < 0 or s_top < 0:
                continue
            fused = w_tfidf * s_lex + w_topic * s_top
            fused_scores[idx] = (fused, s_lex, s_top)

        if not fused_scores:
            return []

        # Sort top-K
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        results: List[RecommendationResult] = []
        for idx, (fused, s_lex, s_top) in sorted_candidates:
            target_pid = self.tfidf.idx_to_id[idx]
            target_meta = self._get_paper_metadata(target_pid)
            dominant_topic = self.topic.get_paper_dominant_topic(target_pid)
            overlapping = self.tfidf.get_overlapping_terms(query_vec, idx, top_n=4)

            terms_str = ", ".join([f"'{t}'" for t, _ in overlapping[:3]])
            explanation = (
                f"Hybrid relevance {fused:.4f} (TF-IDF: {s_lex:.4f}, Topic: {s_top:.4f}). "
                f"Topic: {dominant_topic['label']}. Shared concepts: {terms_str}."
            )

            res_dict = {
                "paper_id": target_pid,
                "title": target_meta["title"],
                "category": target_meta["category"],
                "published_year": target_meta["published_year"],
                "authors": target_meta["authors"],
                "final_score": round(fused, 4),
                "similarity_score": round(fused, 4),
                "tfidf_score": round(s_lex, 4),
                "topic_score": round(s_top, 4),
                "tfidf_contribution": round(w_tfidf * s_lex, 4),
                "topic_contribution": round(w_topic * s_top, 4),
                "dominant_topic": dominant_topic["label"],
                "overlapping_terms": overlapping,
                "explanation": explanation,
            }
            results.append(RecommendationResult(res_dict))

        return results

    def recommend_for_query(
        self,
        query_text: str,
        top_k: int = 10,
        weight_tfidf: float = 0.6,
        weight_topic: float = 0.4,
        category: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        candidate_pool_size: int = 400,
    ) -> List[RecommendationResult]:
        """
        Generates hybrid recommendations for an arbitrary natural language query.
        Projects query into both TF-IDF vocabulary space and LDA topic space.
        """
        clean_text = normalize_text(query_text)
        if not clean_text:
            return []

        query_vec = self.tfidf.vectorizer.transform([clean_text])
        if query_vec.nnz == 0:
            return []

        # Normalize fusion weights
        total_w = weight_tfidf + weight_topic
        if total_w <= 0:
            weight_tfidf, weight_topic = 0.5, 0.5
            total_w = 1.0
        w_tfidf = weight_tfidf / total_w
        w_topic = weight_topic / total_w

        # 1. TF-IDF scores
        tfidf_scores = self.tfidf.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # 2. Topic scores
        query_topic_vec = self.topic.transform_text_query(query_vec)
        topic_scores = self.topic.compute_topic_similarities(query_topic_vec)

        # 3. Apply filters
        self.tfidf._apply_filters(tfidf_scores, category=category, year_min=year_min, year_max=year_max)
        self.tfidf._apply_filters(topic_scores, category=category, year_min=year_min, year_max=year_max)

        valid_mask = (tfidf_scores > 0)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        candidate_k = min(candidate_pool_size, len(valid_indices))
        pool_indices = np.argpartition(tfidf_scores, -candidate_k)[-candidate_k:]

        fused_scores = {}
        for idx in pool_indices:
            s_lex = float(tfidf_scores[idx])
            s_top = float(topic_scores[idx])
            if s_lex <= 0:
                continue
            fused = w_tfidf * s_lex + w_topic * s_top
            fused_scores[idx] = (fused, s_lex, s_top)

        if not fused_scores:
            return []

        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        results: List[RecommendationResult] = []
        for idx, (fused, s_lex, s_top) in sorted_candidates:
            target_pid = self.tfidf.idx_to_id[idx]
            target_meta = self._get_paper_metadata(target_pid)
            dominant_topic = self.topic.get_paper_dominant_topic(target_pid)
            overlapping = self.tfidf.get_overlapping_terms(query_vec, idx, top_n=4)

            terms_str = ", ".join([f"'{t}'" for t, _ in overlapping[:3]])
            explanation = (
                f"Hybrid relevance {fused:.4f} (TF-IDF: {s_lex:.4f}, Topic: {s_top:.4f}). "
                f"Topic: {dominant_topic['label']}. Matched query terms: {terms_str}."
            )

            res_dict = {
                "paper_id": target_pid,
                "title": target_meta["title"],
                "category": target_meta["category"],
                "published_year": target_meta["published_year"],
                "authors": target_meta["authors"],
                "final_score": round(fused, 4),
                "similarity_score": round(fused, 4),
                "tfidf_score": round(s_lex, 4),
                "topic_score": round(s_top, 4),
                "tfidf_contribution": round(w_tfidf * s_lex, 4),
                "topic_contribution": round(w_topic * s_top, 4),
                "dominant_topic": dominant_topic["label"],
                "overlapping_terms": overlapping,
                "explanation": explanation,
            }
            results.append(RecommendationResult(res_dict))

        return results

    def compare_lexical_vs_hybrid(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs both pure lexical search (weight_tfidf=1.0, weight_topic=0.0)
        and hybrid discovery (weight_tfidf=0.5, weight_topic=0.5) to analyze
        rank shifts and semantic broadening.
        """
        lexical = self.recommend_for_query(query_text, top_k=top_k, weight_tfidf=1.0, weight_topic=0.0)
        hybrid = self.recommend_for_query(query_text, top_k=top_k, weight_tfidf=0.5, weight_topic=0.5)

        lexical_ids = [r["paper_id"] for r in lexical]
        hybrid_ids = [r["paper_id"] for r in hybrid]

        common_ids = list(set(lexical_ids) & set(hybrid_ids))
        rank_shifts = {}
        for pid in common_ids:
            lex_rank = lexical_ids.index(pid) + 1
            hyb_rank = hybrid_ids.index(pid) + 1
            rank_shifts[pid] = hyb_rank - lex_rank

        return {
            "query": query_text,
            "lexical_top_k": lexical,
            "hybrid_top_k": hybrid,
            "common_count": len(common_ids),
            "common_ids": common_ids,
            "rank_shifts": rank_shifts,
        }
