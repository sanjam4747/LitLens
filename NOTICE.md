# Attribution and Component Lineage Notice

**Project**: LitLens — NLP-Based Research Paper Recommendation and Discovery System  
**Reference Repository**: [BookMind — Book Recommendation System](https://github.com/NumiKun/Book-Recommendation-System)  
**Original Author**: NumiKun (MIT License, Copyright 2026)

---

## 1. Summary of Attribution

LitLens adapts structural patterns, UI layouts, and content-based recommendation concepts from the open-source BookMind project under the terms of the MIT License.

## 2. Component Lineage Breakdown

### A. Dashboard & UI Patterns (Adapted)
- **Visual Design & Dark Theme**: Custom CSS classes (`.hero-banner`, `.metric-card`, dark mode palette `#0d1117`, card hover animations) adapted into modular Streamlit components.
- **Card-Based Result Presentation**: Adapted from `render_book_card()` / `display_book_grid()` into paper-focused research cards displaying paper title, authors, publication date, categories/subjects, abstract snippet, and similarity score badges.
- **Plotly Visualizations**: Modern chart styling dictionary (`CHART_LAYOUT`), customized color ramps, and interactive horizontal bar charts for similarity rankings and score breakdowns.
- **Multi-Arm Score Fusion**: Adapted the slider-based interactive weight fusion interface from BookMind's Hybrid page to combine TF-IDF cosine similarity, topic model proximity, and semantic similarity.

### B. NLP & Recommendation Patterns (Reused / Adapted)
- **Metadata Soup Concept**: Adapted from BookMind's title + author concatenation to scientific paper attributes (`title + summary/abstract + category + authors`).
- **TF-IDF & Cosine Similarity Engine**: Vectorization with `scikit-learn`'s `TfidfVectorizer` and sparse matrix storage (`scipy.sparse.save_npz` / `.load_npz`).
- **Artifact Serialization**: Parquet metadata caching and sparse matrix storage patterns for low memory footprint and high retrieval speed.

### C. Components Discarded / Explicitly Excluded
- **Collaborative Filtering**: ALS Implicit Matrix Factorization (`implicit`) and SVD++ explicit rating predictor (`scikit-surprise`).
- **Rating Datasets & User Logic**: Book-Crossing `Ratings.csv` and `Users.csv`, rating distribution visualizations, user history tracking, and user encoders.
- **SQL Ingestion Scripts**: SQL schemas and insertion scripts tailored specifically for the Book-Crossing schema.

