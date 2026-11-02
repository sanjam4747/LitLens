# LitLens Phase 3: Recommendation Engine Evaluation Report

## 1. Overview & System Performance
- **Indexed Document Count:** 287,421 arXiv research papers
- **Engine Architecture:** Two-arm score fusion (Sparse TF-IDF Cosine + LDA Topic Distribution Alignment)
- **Average Retrieval Latency:** 2517.77 ms per query (across all 287k papers)
- **Engine Process Memory:** 1209.13 MB RAM
- **Pairwise Matrix Generation:** None (Single-query sparse dot products used exclusively)

---

## 2. Topic Modeling Configuration & Discovered Themes
- **Model:** Scikit-learn `LatentDirichletAllocation` (Variational Inference, Online learning)
- **Topic Count ($K$):** 16 topics
- **Vocabulary Filtering:** Pruned corpus-ubiquitous terms (> 25% document frequency) to preserve topic discriminability.
- **Top Discovered Themes:**
  1. *Large Language Models & Multi-Agent Reasoning* (`llms, large language, reasoning, agents`)
  2. *Computer Vision & Object Detection* (`image, pattern recognition, vision pattern, detection, visual`)
  3. *Natural Language Processing & Computation* (`natural language, language processing, computation language`)
  4. *Reinforcement Learning & Control* (`reinforcement learning, policy, continuous, systems, intelligence`)
  5. *Deep Neural Networks & Architectures* (`networks, neural networks, deep, convolutional, optimization`)

---

## 3. Detailed Query Evaluations across 5 Domains

### Query 1: Computer Vision
* **Natural-Language Query:** `"deep residual networks for image recognition and feature extraction"`
* **Filters Applied:** None (Cross-disciplinary search)
* **Latency:** Lexical: 3586.24 ms | Hybrid: 2387.67 ms

#### Top Hybrid Recommendations:
| Rank | arXiv ID | Paper Title | Category | Year | Final Score | TF-IDF | Topic | Dominant Topic | Key Shared Concepts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `abs-1605.07146v4` | Wide Residual Networks | Computer Vision and Pattern Recognition | 2016 | **0.5997** | 0.3389 | 0.9909 | Image / Pattern Recognition / Vision Pattern | `residual networks`, `deep residual`, `residual` |
| #2 | `abs-1608.05895v1` | VoxResNet: Deep Voxelwise Residual Networks for ... | Computer Vision and Pattern Recognition | 2016 | **0.5797** | 0.3055 | 0.9911 | Image / Pattern Recognition / Vision Pattern | `deep residual`, `residual networks`, `image recognition` |
| #3 | `abs-1609.05119v1` | Deep Impression: Audiovisual Deep Residual Netwo... | Computer Vision and Pattern Recognition | 2016 | **0.5767** | 0.2975 | 0.9955 | Image / Pattern Recognition / Vision Pattern | `deep residual`, `residual networks`, `residual` |
| #4 | `abs-1609.05672v4` | Multi-Residual Networks: Improving the Speed and... | Computer Vision and Pattern Recognition | 2016 | **0.5724** | 0.2933 | 0.9909 | Image / Pattern Recognition / Vision Pattern | `residual networks`, `deep residual`, `residual` |
| #5 | `abs-1707.02921v1` | Enhanced Deep Residual Networks for Single Image... | Computer Vision and Pattern Recognition | 2017 | **0.5721** | 0.2924 | 0.9915 | Image / Pattern Recognition / Vision Pattern | `residual networks`, `deep residual`, `residual` |

### Query 2: Natural Language Processing / LLMs
* **Natural-Language Query:** `"prompt engineering in large language models for zero-shot reasoning"`
* **Filters Applied:** Category = 'Computation and Language (Natural Language Processing)'
* **Latency:** Lexical: 2456.39 ms | Hybrid: 2536.82 ms

#### Top Hybrid Recommendations:
| Rank | arXiv ID | Paper Title | Category | Year | Final Score | TF-IDF | Topic | Dominant Topic | Key Shared Concepts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `abs-2407.12865v1` | GRAD-SUM: Leveraging Gradient Summarization for ... | Computation and Language (Natural Language Processing) | 2024 | **0.5787** | 0.3373 | 0.9408 | Natural Language / Natural / Language Processing | `prompt engineering`, `prompt`, `engineering large` |
| #2 | `abs-2310.01691v2` | Zero-Shot Continuous Prompt Transfer: Generalizi... | Computation and Language (Natural Language Processing) | 2023 | **0.5699** | 0.2848 | 0.9975 | Natural Language / Natural / Language Processing | `models zero-shot`, `prompt`, `zero-shot` |
| #3 | `abs-2505.11665v1` | Multilingual Prompt Engineering in Large Languag... | Computation and Language (Natural Language Processing) | 2025 | **0.5673** | 0.2811 | 0.9965 | Natural Language / Natural / Language Processing | `engineering large`, `prompt engineering`, `prompt` |
| #4 | `abs-2303.07142v3` | Large Language Models in the Workplace: A Case S... | Computation and Language (Natural Language Processing) | 2023 | **0.5523** | 0.2557 | 0.9972 | Natural Language / Natural / Language Processing | `prompt engineering`, `prompt`, `engineering` |
| #5 | `abs-2311.05169v1` | Large Language Models and Prompt Engineering for... | Computation and Language (Natural Language Processing) | 2023 | **0.5522** | 0.2551 | 0.9979 | Natural Language / Natural / Language Processing | `prompt engineering`, `prompt`, `engineering` |

### Query 3: Reinforcement Learning & Robotics
* **Natural-Language Query:** `"reinforcement learning for robotic manipulation and continuous control"`
* **Filters Applied:** None (Cross-disciplinary search)
* **Latency:** Lexical: 2418.45 ms | Hybrid: 2353.82 ms

#### Top Hybrid Recommendations:
| Rank | arXiv ID | Paper Title | Category | Year | Final Score | TF-IDF | Topic | Dominant Topic | Key Shared Concepts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `abs-2205.06311v1` | Provably Safe Deep Reinforcement Learning for Ro... | Robotics | 2022 | **0.6381** | 0.3987 | 0.9973 | Reinforcement / Reinforcement Learning / Neural | `learning robotic`, `robotic manipulation`, `robotic` |
| #2 | `abs-2604.01142v1` | Deep Reinforcement Learning for Robotic Manipula... | Robotics | 2026 | **0.6303** | 0.3860 | 0.9968 | Reinforcement / Reinforcement Learning / Neural | `learning robotic`, `robotic manipulation`, `robotic` |
| #3 | `abs-1803.06773v1` | Composable Deep Reinforcement Learning for Robot... | Machine Learning | 2018 | **0.6274** | 0.3810 | 0.9968 | Reinforcement / Reinforcement Learning / Neural | `learning robotic`, `robotic manipulation`, `robotic` |
| #4 | `abs-2206.06282v2` | Analysis of Randomization Effects on Sim2Real Tr... | Robotics | 2022 | **0.6260** | 0.3782 | 0.9977 | Reinforcement / Reinforcement Learning / Neural | `learning robotic`, `robotic manipulation`, `robotic` |
| #5 | `abs-2107.13356v1` | Value-Based Reinforcement Learning for Continuou... | Robotics | 2021 | **0.6152** | 0.3609 | 0.9967 | Reinforcement / Reinforcement Learning / Neural | `robotic manipulation`, `continuous control`, `robotic` |

### Query 4: Graph Machine Learning
* **Natural-Language Query:** `"graph convolutional networks for node classification and link prediction"`
* **Filters Applied:** None (Cross-disciplinary search)
* **Latency:** Lexical: 2350.29 ms | Hybrid: 2377.55 ms

#### Top Hybrid Recommendations:
| Rank | arXiv ID | Paper Title | Category | Year | Final Score | TF-IDF | Topic | Dominant Topic | Key Shared Concepts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `abs-2106.07297v2` | Node Classification Meets Link Prediction on Kno... | Machine Learning | 2021 | **0.6902** | 0.4840 | 0.9994 | Networks / Neural / Algorithm | `node classification`, `link prediction`, `classification link` |
| #2 | `abs-2307.00865v1` | A Survey on Graph Classification and Link Predic... | Machine Learning | 2023 | **0.6873** | 0.4792 | 0.9995 | Networks / Neural / Algorithm | `classification link`, `link prediction`, `graph convolutional` |
| #3 | `abs-2107.10015v1` | Relational Graph Convolutional Networks: A Close... | Machine Learning | 2021 | **0.6656** | 0.4430 | 0.9995 | Networks / Neural / Algorithm | `graph convolutional`, `convolutional networks`, `classification link` |
| #4 | `abs-1911.03082v2` | Composition-based Multi-Relational Graph Convolu... | Machine Learning | 2019 | **0.6534** | 0.4228 | 0.9993 | Networks / Neural / Algorithm | `graph convolutional`, `convolutional networks`, `classification link` |
| #5 | `abs-2408.01408v1` | Derivation of Back-propagation for Graph Convolu... | Machine Learning | 2024 | **0.6515** | 0.4197 | 0.9991 | Networks / Neural / Algorithm | `graph convolutional`, `convolutional networks`, `node classification` |

### Query 5: Adversarial Robustness & Security
* **Natural-Language Query:** `"adversarial attacks and defense robustness in deep neural networks"`
* **Filters Applied:** None (Cross-disciplinary search)
* **Latency:** Lexical: 2344.77 ms | Hybrid: 2365.69 ms

#### Top Hybrid Recommendations:
| Rank | arXiv ID | Paper Title | Category | Year | Final Score | TF-IDF | Topic | Dominant Topic | Key Shared Concepts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | `abs-1902.11029v1` | Enhancing the Robustness of Deep Neural Networks... | Machine Learning | 2019 | **0.6569** | 0.4288 | 0.9989 | Networks / Neural / Algorithm | `robustness deep`, `defense`, `adversarial attacks` |
| #2 | `abs-1903.10033v1` | A Formalization of Robustness for Deep Neural Ne... | Machine Learning | 2019 | **0.6445** | 0.4080 | 0.9994 | Networks / Neural / Algorithm | `robustness deep`, `deep neural`, `robustness` |
| #3 | `abs-2011.01539v1` | Recent Advances in Understanding Adversarial Rob... | Machine Learning | 2020 | **0.6407** | 0.4020 | 0.9989 | Networks / Neural / Algorithm | `robustness deep`, `adversarial`, `deep neural` |
| #4 | `abs-1802.07124v3` | Out-distribution training confers robustness to ... | Machine Learning | 2018 | **0.6251** | 0.3756 | 0.9992 | Networks / Neural / Algorithm | `robustness deep`, `deep neural`, `robustness` |
| #5 | `abs-2102.09012v1` | Improving Hierarchical Adversarial Robustness of... | Machine Learning | 2021 | **0.6195** | 0.3972 | 0.9530 | Networks / Neural / Algorithm | `robustness deep`, `adversarial`, `deep neural` |

## 4. Qualitative Relevance Comparison: Lexical vs. Hybrid

1. **Lexical Retrieval (TF-IDF Only, $w_{\text{tfidf}}=1.0, w_{\text{topic}}=0.0$):**
   * *Strengths:* Exceptional precision when user provides exact technical terminology (e.g., exact matches on `"residual networks"`, `"convolutional"`).
   * *Limitations:* Ranks purely by term frequency and IDF. Papers with sparse abstracts or slightly different phrasing miss out on ranking near the top.

2. **Hybrid Retrieval (Fused TF-IDF + LDA, $w_{\text{tfidf}}=0.6, w_{\text{topic}}=0.4$):**
   * *Strengths:* Balances specific keyword matching with broader thematic alignment. Papers that address the exact theoretical topic (e.g. computer vision feature extraction) receive a boost from the topic arm, ensuring that semantically relevant research in the same paradigm is promoted.
   * *Interpretability:* Component contributions ($w_1 \cdot s_{\text{tfidf}}$ and $w_2 \cdot s_{\text{topic}}$) and term overlap are transparently exposed for every result.
