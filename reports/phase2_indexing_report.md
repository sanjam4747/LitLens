# LitLens Phase 2: Dataset Processing & Core NLP Indexing Report

## 1. Executive Summary
LitLens Phase 2 establishes the data ingestion, quality validation, domain-aware NLP text preprocessing, and sparse TF-IDF retrieval index for **287,421 research papers** from the arXiv scientific corpus.

All artifacts were generated reproducibly without altering or overwriting the original dataset (`data/arXiv_scientific_dataset.csv`). The index operates strictly in sparse compressed format (`scipy.sparse.csr_matrix`), achieving single-query cosine similarity retrieval in sub-second latency (< 100 ms) while completely avoiding dense matrix allocation.

---

## 2. Dataset Statistics & Validation (Task 1)

* **Source File:** `data/arXiv_scientific_dataset.csv` (Preserved intact, unmodified)
* **Total Records:** **287,421** papers
* **Canonical Columns (10/10 Verified):** `id`, `title`, `category`, `category_code`, `published_date`, `updated_date`, `authors`, `first_author`, `summary`, `summary_word_count`
* **Completeness:** 0 missing values across all columns (100% complete)
* **Primary Key Uniqueness:** 287,421 unique IDs (0 duplicates)
* **Base Paper Identification:** 287,395 unique base IDs (26 papers have revisions across versions v1–v3)
* **Temporal Coverage:** August 1, 1993 to June 18, 2026 (0 date parse failures)
* **Abstract Statistics:**
  * Mean length: 170.96 words
  * Median length: 170.00 words
  * Range: 1 to 552 words
* **Top Subject Categories:**
  1. *Machine Learning* (`cs.LG`): 80,452 papers (27.99%)
  2. *Computer Vision and Pattern Recognition* (`cs.CV`): 71,106 papers (24.74%)
  3. *Computation and Language / NLP* (`cs.CL`): 37,251 papers (12.96%)
  4. *Artificial Intelligence* (`cs.AI`): 28,726 papers (9.99%)
  5. *Machine Learning / Statistics* (`stat.ML`): 13,892 papers (4.83%)

---

## 3. Preprocessing Decisions (Task 2)

### A. Scientific Terminology & Acronym Preservation
Standard stopword strippers and punctuation tokenizers often destroy critical scientific expressions. In LitLens:
1. **Hyphenated and Alphanumeric Technical Terms:** Tokens like `resnet-50`, `3d-cnn`, `bert-large`, `l1-regularized`, `quasi-convex`, `k-means`, `o(n)` are preserved using a custom regex token pattern:
   $$\text{Pattern: } \texttt{(?u)\textbackslash b[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*\textbackslash b}$$
2. **LaTeX and Mathematical Formatting:** LaTeX structural markup (e.g., `\textbf{...}`, `\emph{...}`, `\cite{...}`) is flattened to extract the underlying natural language tokens while removing markup noise. Math delimiters (`$...$`, `$$...$$`) are cleanly stripped.
3. **HTML / Entity Decoding:** Encoded entities (e.g. `&amp;`, `&lt;`, `&gt;`) are unescaped prior to normalization.

### B. Conservative Stopword Filtering
Standard English stopwords from Scikit-Learn (`'english'`) are applied to eliminate grammatical syntax without discarding technical verbs and descriptors.

### C. Schema Standardization & Parquet Storage
A standardized catalog is exported to `models/papers_catalog.parquet` using `pyarrow` and snappy compression:
* Retains original `title`, `summary`, `authors`, and dates for faithful display in the dashboard.
* Adds parsed publication timestamps, year integers, parsed author display strings, and `is_latest_version` flags.

---

## 4. Feature Engineering & TF-IDF Mathematical Model (Task 3 & 4)

### A. Weighted Representation ("Soup")
Titles convey the highest information density and core focus of research papers. To reflect this in vector space, title terms are repeated with a title weight multiplier ($w_{\text{title}} = 3$), combined with explicit category tags and abstract text:
$$\text{Soup}(d) = \underbrace{(Title(d) \oplus \dots \oplus Title(d))}_{3\times} \oplus Category(d) \oplus Summary(d)$$

### B. Mathematical TF-IDF Formulation
For term $t$ in document $d$ within corpus $D$:
1. **Sublinear Term Frequency:** Abstracts often repeat keywords; log-scaling prevents high term frequency from dominating:
   $$\text{tf}_{\text{sub}}(t, d) = 1 + \ln(\text{tf}(t, d)) \quad \text{for } \text{tf}(t, d) > 0$$
2. **Inverse Document Frequency with Smooth Weighting:**
   $$\text{idf}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{df}(t, D)}\right) + 1$$
3. **Raw TF-IDF Weight:**
   $$w(t, d) = \text{tf}_{\text{sub}}(t, d) \times \text{idf}(t, D)$$
4. **$L_2$ Normalization:** Each document vector $\vec{v}_d$ is normalized onto the unit hypersphere:
   $$\vec{u}_d = \frac{\vec{v}_d}{\|\vec{v}_d\|_2}$$
   This ensures that the Cosine Similarity between document $i$ and document $j$ is equivalent to their dot product:
   $$\text{CosineSimilarity}(\vec{u}_i, \vec{u}_j) = \vec{u}_i \cdot \vec{u}_j$$

### C. Vectorizer Configuration
* **Vocabulary Cap (`max_features`):** 50,000 top n-grams
* **N-gram Range:** $(1, 2)$ — Unigrams and Bigrams (capturing phrases like `neural network`, `transfer learning`, `gradient descent`)
* **Document Frequency Cutoffs:** $\text{min\_df} = 5$ (prunes single-occurrence typos), $\text{max\_df} = 0.8$ (prunes non-informative corpus-wide terms)
* **Numeric Type:** `np.float32` (halves memory overhead relative to `float64`)
* **Sparse Format:** Compressed Sparse Row (`scipy.sparse.csr_matrix`)

---

## 5. Artifact Sizes, Memory & Execution Time

| Artifact / Metric | Measurement |
| :--- | :--- |
| **Catalog Parquet (`papers_catalog.parquet`)** | 238.38 MB |
| **TF-IDF Sparse Matrix (`tfidf_matrix.npz`)** | 178.04 MB (Compressed CSR) |
| **Fitted Vectorizer (`tfidf_vectorizer.pkl`)** | 1.89 MB |
| **Metadata Index (`tfidf_metadata.pkl`)** | 9.06 MB |
| **Sparse Matrix Shape** | $(287,421 \times 50,000)$ |
| **Non-Zero Entries ($\text{nnz}$)** | 33,105,591 entries |
| **Matrix Density** | $0.002304$ ($0.23\%$ non-zero) |
| **Dense vs Sparse Comparison** | Sparse: 178 MB vs. Dense: 57.48 GB (**99.69% space saving**) |
| **Total Ingestion & Vectorization Runtime** | 215.03 seconds (~3.58 minutes) |
| **Peak Pipeline Memory Consumption** | ~2.65 GB RAM |
| **Single-Query Cosine Similarity Latency** | 25–45 ms |

---

## 6. Test Suite & Verification Results

19 automated tests were implemented and verified with a 100% pass rate:
* **Unit Tests (`tests/test_data_loader.py`):** Schema validation, missing columns detection, base ID parsing, version extraction, and quality report generation.
* **Unit Tests (`tests/test_preprocessing.py`):** LaTeX/HTML stripping, technical term retention, metadata standardization, weighted soup frequency verification, Parquet roundtrip.
* **Unit Tests (`tests/test_tfidf_recommender.py`):** Unigram/bigram vocabulary formation, sparse matrix format, artifact serialization/deserialization, paper-to-paper recommendation, and text query matching.
* **Integration Tests (`tests/test_pipeline_integration.py`):** Verified parity across the 287,421 documents, row alignment between catalog and sparse matrix, sub-100ms query latency, and self-exclusion correctness.

```text
Ran 19 tests in 5.512s
OK
```

---

## 7. Known Limitations & Phase 3 Roadmap

1. **Vocabulary Out-of-Vocabulary (OOV):** TF-IDF cannot capture synonyms or conceptual paraphrasing if words do not overlap (e.g., "deep learning" vs "neural representations" when unshared). This will be resolved in Phase 3/4 through Topic Modeling (LDA/NMF) and Dense Semantic Embeddings.
2. **Fixed Vocabulary Cap (50,000 terms):** Extremely rare scientific compound names appearing in fewer than 5 documents are omitted to preserve low latency and memory efficiency.
3. **Cross-listing Discrepancies:** A minority of papers (122 titles) appear in multiple categories or revisions. The `is_latest_version` flag and `base_id` allow filtering or presenting deduplicated papers during discovery.

