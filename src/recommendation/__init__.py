"""
LitLens recommendation engines package.
Contains TF-IDF Recommender, Topic Modeler (LDA), and Hybrid Recommender.
"""

from src.recommendation.tfidf_recommender import TFIDFRecommender
from src.recommendation.topic_modeler import TopicModeler
from src.recommendation.hybrid_engine import HybridRecommender

__all__ = ["TFIDFRecommender", "TopicModeler", "HybridRecommender"]
