# TF-IDF Document Retrieval — Approach & Study Guide

**Module:** `04_tfidf_retrieval.py`  
**Project:** Contradiction-Aware RAG  
**Phase:** 2 — Retrieval Layer  

---

## Table of Contents

1. [What Is TF-IDF?](#1-what-is-tf-idf)
2. [The Math](#2-the-math)
3. [How the Retriever Works (Step-by-Step)](#3-how-the-retriever-works-step-by-step)
4. [Implementation Choices & Why](#4-implementation-choices--why)
5. [Strengths of TF-IDF Retrieval](#5-strengths-of-tf-idf-retrieval)
6. [Weaknesses & Failure Cases](#6-weaknesses--failure-cases)
7. [Evaluation Metrics to Know](#7-evaluation-metrics-to-know)
8. [Comparison: TF-IDF vs. Other Retrieval Methods](#8-comparison-tf-idf-vs-other-retrieval-methods)
9. [Key Papers & Resources to Study](#9-key-papers--resources-to-study)
10. [What to Study Next (Roadmap)](#10-what-to-study-next-roadmap)

---

## 1. What Is TF-IDF?

**TF-IDF** (Term Frequency–Inverse Document Frequency) is a classical, statistics-based method to measure how *important* a word is to a specific document within a larger collection (corpus).

The core intuition is:
- A word that appears **often in one document** (`↑ TF`) is likely important to that document.
- A word that appears **across all documents** (e.g., "the", "and") is **not discriminative** (`↓ IDF`).
- The product of these two penalizes common words and rewards rare, informative ones.

### Role in Retrieval

In a retrieval pipeline:
1. Every document is represented as a **TF-IDF vector** (one dimension per vocabulary term).
2. A user query is transformed into the **same vector space**.
3. Documents are ranked by **cosine similarity** between their vector and the query vector.

This gives a relevance score: documents sharing rare, specific terms with the query rank higher.

---

## 2. The Math

### 2.1 Term Frequency (TF)

Measures how often term `t` appears in document `d`:

```
TF(t, d) = count(t in d) / total_terms(d)
```

**Sublinear TF** (used in our implementation) dampens the effect of very high raw counts:

```
TF_sublinear(t, d) = 1 + log(count(t in d))   if count > 0, else 0
```

> Why? A word appearing 100 times isn't 10× more informative than one appearing 10 times.
> Sublinear scaling shrinks this gap.

---

### 2.2 Inverse Document Frequency (IDF)

Penalizes terms that appear in many documents (i.e., not discriminative):

```
IDF(t, D) = log( (1 + N) / (1 + DF(t)) ) + 1
```

Where:
- `N` = total number of documents in corpus
- `DF(t)` = number of documents containing term `t`
- `+1` smoothing prevents division by zero

> A term appearing in every document (DF = N) gets IDF ≈ 1 (no boost).  
> A very rare term (DF = 1) gets a high IDF ≈ log(N) + 1.

---

### 2.3 TF-IDF Score

```
TFIDF(t, d, D) = TF(t, d) × IDF(t, D)
```

Each document `d` becomes a **sparse vector** in ℝ^|V| (vocabulary size):

```
d_vec = [ TFIDF(t1, d), TFIDF(t2, d), ..., TFIDF(t|V|, d) ]
```

---

### 2.4 Cosine Similarity

Given query vector **q** and document vector **d**:

```
cosine_similarity(q, d) = (q · d) / (||q|| × ||d||)
```

- Ranges from **0** (completely unrelated) to **1** (identical direction).
- Normalizes for document length, so long documents aren't artificially boosted.

---

### 2.5 N-grams

Instead of only single words (unigrams), we also index **bigrams** (pairs of consecutive words):

```
"hurricane flood" → treated as a single token alongside "hurricane" and "flood"
```

This captures phrases, improving precision for multi-word queries.

---

## 3. How the Retriever Works (Step-by-Step)

```
CORPUS (5,754 Multi-News articles)
        │
        ▼
[ Tokenization + Stop Word Removal ]
   "The hurricane caused flooding" → ["hurricane", "caused", "flooding",
                                       "hurricane caused", "caused flooding"]
        │
        ▼
[ TF-IDF Vectorization ]
   Each article → sparse vector of 50,000 TF-IDF weights
        │
        ▼
[ Sparse Document-Term Matrix ]
   Shape: (5754 docs × 50000 terms)
        │
  ──────┴──────── (index built once, reused for all queries)
        │
        ▼
[ User Query ]
   "hurricane flood casualties"
        │
        ▼
[ Transform Query → TF-IDF Vector ]
   Same vocabulary, same IDF weights
        │
        ▼
[ Cosine Similarity vs. All Documents ]
   scores = query_vec · matrix.T   → array of 5754 scores
        │
        ▼
[ Top-K Ranking ]
   argsort(scores)[::-1][:k] → indices of top-k documents
        │
        ▼
[ Return Results ]
   rank, doc_id, cluster_id, score, snippet, summary
```

---

## 4. Implementation Choices & Why

| Parameter | Value Used | Rationale |
|:---|:---|:---|
| `sublinear_tf = True` | `1 + log(tf)` | Prevents long news articles from dominating simply due to word repetition |
| `ngram_range = (1, 2)` | Unigrams + Bigrams | Captures phrases like "gun control", "death toll", "climate change" |
| `max_features = 50,000` | Top 50k terms | Balances coverage vs. memory; rare misspellings and names capped |
| `min_df = 2` | Term must appear in ≥2 docs | Removes hapax legomena (unique spelling variants, OCR artifacts) |
| `stop_words = "english"` | 318 common words removed | "the", "is", "a" carry no retrieval signal |
| `strip_accents = "unicode"` | Normalize accents | "naïve" and "naive" treated identically |
| `token_pattern` | `[a-zA-Z][a-zA-Z0-9]{1,}` | Ignores pure numbers, single characters, punctuation artifacts |
| Cosine similarity | sklearn `cosine_similarity` | Length-normalized; robust to document size variance |

---

## 5. Strengths of TF-IDF Retrieval

1. **Interpretable:** You can inspect which terms drove the score. No black-box neural weights.
2. **Fast:** Vectorization is a one-time `O(N × V)` operation. Query-time is a sparse dot product — milliseconds.
3. **No training data needed:** Purely unsupervised. Works on any domain out of the box.
4. **Exact term matching:** If a query contains a precise term (e.g., a name, a specific place), TF-IDF will reliably surface documents containing it.
5. **Low memory (sparse):** The document-term matrix is stored as a sparse CSR matrix — only non-zero entries stored.
6. **Strong baseline:** In IR benchmarks, TF-IDF is often within 5–10% of neural retrievers on keyword-heavy queries.

---

## 6. Weaknesses & Failure Cases

| Failure Mode | Example | Root Cause |
|:---|:---|:---|
| **Vocabulary mismatch** | Query: "car crash" — Document says "automobile accident" | No synonym resolution; operates on exact tokens |
| **Semantic gap** | Query: "economic downturn" — Document discusses "GDP contraction" | No concept-level understanding |
| **Short query failure** | Query: "war" — Too common, IDF nearly zero for all docs | Single-term queries have low discriminative power |
| **Long document bias** | Without sublinear_tf, 1000-word docs score higher than 100-word ones | Mitigated by sublinear TF and cosine normalization |
| **Context blindness** | "Apple" (fruit) vs "Apple" (company) — no disambiguation | Bag-of-words: word order and context ignored |
| **Negation blindness** | "not guilty" may rank documents about "guilty" verdicts | Negation words typically stop-worded out |

---

## 7. Evaluation Metrics to Know

When evaluating a retrieval system like this, these are the standard IR metrics:

### Precision@K
> Of the top-K retrieved documents, what fraction are actually relevant?

```
P@K = (# relevant docs in top-K) / K
```

### Recall@K
> Of all relevant documents in the corpus, what fraction appear in the top-K?

```
R@K = (# relevant docs in top-K) / (total relevant docs)
```

### Mean Average Precision (MAP)
> Averages precision at each rank where a relevant doc is found, over all queries.
> Rewards retrieving relevant docs early (high rank).

### NDCG (Normalized Discounted Cumulative Gain)
> Like MAP but supports graded relevance (e.g., "highly relevant" vs. "somewhat relevant").

```
DCG@K = Σ (rel_i / log2(i+1))  for i = 1..K
NDCG@K = DCG@K / IDCG@K        (IDCG = ideal/perfect ranking)
```

### MRR (Mean Reciprocal Rank)
> For each query, what is the rank of the first correct result?

```
MRR = (1/|Q|) × Σ (1 / rank_i)
```

---

## 8. Comparison: TF-IDF vs. Other Retrieval Methods

| Method | Representation | Handles Synonyms | Speed | Needs Training | Best For |
|:---|:---|:---|:---|:---|:---|
| **TF-IDF** (this module) | Sparse bag-of-words | ❌ No | ⚡ Very fast | ❌ No | Keyword-heavy queries, baseline |
| **BM25** | Sparse, probabilistic | ❌ No | ⚡ Very fast | ❌ No | Better TF saturation, industry standard |
| **Dense Retrieval** (DPR, bi-encoder) | Dense neural embeddings | ✅ Yes | 🐢 Slower (ANN search) | ✅ Yes | Semantic/paraphrase queries |
| **ColBERT** | Token-level dense | ✅ Yes | 🐢 Slower | ✅ Yes | High precision + recall |
| **Hybrid (TF-IDF + Dense)** | Both | ✅ Partial | Medium | ✅ Yes | Best of both worlds |
| **Cross-encoder re-ranking** | Full cross-attention | ✅ Yes | 🐌 Very slow | ✅ Yes | Re-ranking top-K only |

> **In this project:** TF-IDF retrieves the top-K candidates → later a cross-encoder NLI model
> (Phase 3) re-ranks and identifies contradictions.

### BM25 vs. TF-IDF

BM25 is a probabilistic extension of TF-IDF with two key improvements:
- **TF saturation parameter** `k1`: prevents raw TF from growing unboundedly
- **Length normalization parameter** `b`: adjusts for document length

```
BM25(t, d) = IDF(t) × [ TF(t,d) × (k1+1) ] / [ TF(t,d) + k1×(1 - b + b×|d|/avgdl) ]
```

BM25 (with `k1=1.5`, `b=0.75`) is typically the strongest sparse retrieval baseline.

---

## 9. Key Papers & Resources to Study

### Foundational IR Theory

| Paper / Book | Why Read It |
|:---|:---|
| **Sparck Jones (1972)** — *A statistical interpretation of term specificity and its application in retrieval* | Original IDF paper. Foundational. |
| **Robertson & Zaragoza (2009)** — *The Probabilistic Relevance Framework: BM25 and Beyond* | The definitive BM25 reference. Understanding this is essential for modern IR. |
| **Manning, Raghavan & Schütze** — *Introduction to Information Retrieval* (free online) | Best textbook for IR fundamentals. Chapters 6–8 cover TF-IDF, vector space model, and evaluation. |

### Dense Retrieval (to study after TF-IDF)

| Paper | Why Read It |
|:---|:---|
| **Karpukhin et al. (2020)** — *Dense Passage Retrieval for Open-Domain QA* (DPR) | The paper that popularized bi-encoder dense retrieval. |
| **Khattab & Zaharia (2020)** — *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction* | State-of-the-art dense retrieval. Token-level matching. |
| **Reimers & Gurevych (2019)** — *Sentence-BERT: Sentence Embeddings using Siamese BERT Networks* | Foundation for all sentence-embedding based retrieval. |

### RAG-Specific

| Paper | Why Read It |
|:---|:---|
| **Lewis et al. (2020)** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | The original RAG paper (Facebook AI). Must-read. |
| **Shi et al. (2023)** — *REPLUG: Retrieval-Augmented Black-Box Language Models* | Shows how retrieval quality directly affects generation quality. |

### Contradiction / NLI (Phase 3 prep)

| Paper | Why Read It |
|:---|:---|
| **Schuster et al. (2021)** — *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence* | The VitaminC dataset used in this project. |
| **He et al. (2021)** — *DeBERTa: Decoding-enhanced BERT with Disentangled Attention* | Best NLI cross-encoder backbone to study. |

---

## 10. What to Study Next (Roadmap)

```
[DONE] Phase 1: EDA & Preprocessing
         └── Understand datasets (Multi-News, WCEP-10, VitaminC)

[DONE] Phase 2a: TF-IDF Retrieval   ← YOU ARE HERE
         └── Understand: TF-IDF math, cosine similarity, n-grams

[ ] Phase 2b: Upgrade to BM25
         └── Study: rank_bm25 library, k1/b parameters
         └── Compare MAP/NDCG vs TF-IDF on a labeled subset

[ ] Phase 2c: Dense Retrieval (optional advanced)
         └── Study: sentence-transformers, FAISS/ChromaDB
         └── Model: all-MiniLM-L6-v2 or bge-small-en-v1.5

[ ] Phase 3: Contradiction Detection
         └── Study: NLI task, cross-encoders, SUPPORTS/REFUTES classification
         └── Model: cross-encoder/nli-deberta-v3-small (HuggingFace)
         └── Dataset: VitaminC (already preprocessed in 03_vitaminc_eda.py)

[ ] Phase 4: Contradiction-Aware RAG
         └── Study: prompt engineering for conflict surfacing
         └── Integrate: Retriever → Contradiction Scorer → LLM synthesis
```

### Immediate study checklist

- [ ] Read **Manning et al. IIR** — Chapter 6 (Scoring, term weighting, and the vector space model)
- [ ] Understand **cosine similarity** geometrically — why it's angle, not magnitude
- [ ] Run the retriever with 5+ diverse queries and observe where it succeeds vs. fails
- [ ] Try changing `ngram_range` to `(1, 1)` — observe precision drop on phrase queries
- [ ] Try lowering `min_df` to 1 — observe vocab explosion and recall change
- [ ] Read about **inverted indexes** — how production search engines (Elasticsearch, Lucene) implement TF-IDF at scale

---

## Quick Reference: Parameters in `04_tfidf_retrieval.py`

```bash
# Usage
python 04_tfidf_retrieval.py [--query QUERY] [--top_k N] [--max_features N] [--sample_size N]

# Examples
python 04_tfidf_retrieval.py --query "election fraud ballot rigging" --top_k 5
python 04_tfidf_retrieval.py --query "climate change flooding" --top_k 10 --max_features 100000

# Interactive multi-query loop
python 04_tfidf_retrieval.py

# Inside interactive mode: override top_k inline
Query> gun control legislation --top_k 8
```

**Output location:** `output/tfidf_index/`
- `tfidf_index_stats.json` — index metadata (vocab size, matrix shape, parameters)
- `results_<query>.jsonl` — ranked retrieval results per query

---

*Part of the Contradiction-Aware RAG project — IRE Course, 2026.*
