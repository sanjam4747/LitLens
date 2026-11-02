"""
LitLens: NLP-Based Research Paper Recommendation and Discovery System
"""

from src.data_loader import load_raw_dataset, inspect_dataset_quality
from src.preprocessing import standardize_metadata, build_weighted_soup, save_processed_catalog, load_processed_catalog
from src.recommendation.tfidf_recommender import TFIDFRecommender

__version__ = "0.1.0"
__all__ = [
    "load_raw_dataset",
    "inspect_dataset_quality",
    "standardize_metadata",
    "build_weighted_soup",
    "save_processed_catalog",
    "load_processed_catalog",
    "TFIDFRecommender",
]
