# Contradiction-Aware RAG

An EDA, preprocessing, retrieval, and dataset feasibility pipeline for building a **Contradiction-Aware Retrieval-Augmented Generation (RAG)** system.

---

## Overview

When standard RAG systems retrieve multiple documents across different news sources covering the same event, they often encounter conflicting or contradictory information (e.g., conflicting casualty counts, dates, or factual claims). This project explores multi-document corpora and fact-verification datasets to evaluate:
1. Suitable multi-document retrieval corpora (Multi-News, WCEP-10).
2. Labeled training/evaluation datasets for contradiction detection (VitaminC).
3. Sparse vs Dense retrieval for finding relevant multi-source documents.

---

## Datasets Explored

### 1. Multi-News (`01_multinews_eda.py`)
- **Paper**: Fabbri et al. (2019) — [arXiv:1906.01749](https://arxiv.org/abs/1906.01749)
- **Source**: `alexfabbri/multi_news` on Hugging Face
- **Description**: Multi-document summarization dataset containing clusters of 2–10+ news articles per event concatenated with ` ||||| `.
- **Role**: Primary document retrieval corpus candidate.
- **Preprocessing**: Splits clusters into standalone article records, normalizes whitespace, cleans editorial formatting, and assigns unique document/cluster IDs and MD5 content hashes.

### 2. WCEP-10 (`02_wcep10_eda.py`)
- **Paper**: Gholipour Ghalandari et al. (2020) — [arXiv:2005.10070](https://arxiv.org/abs/2005.10070)
- **Source**: `ccdv/WCEP-10` on Hugging Face
- **Description**: Multi-document news dataset based on the Wikipedia Current Events Portal (up to 10 articles per cluster separated by `</s>`).
- **Role**: Alternative/development retrieval corpus.
- **Preprocessing**: Splits at `</s>`, filters short snippets (<10 words), deduplicates identical content, and generates standardized document IDs.

### 3. VitaminC (`03_vitaminc_eda.py`)
- **Paper**: Schuster, Fisch & Barzilay (NAACL 2021) — [arXiv:2104.10850](https://arxiv.org/abs/2104.10850)
- **Source**: `tals/vitaminc` on Hugging Face
- **Description**: Fact-verification dataset with `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` labels. Contains contrastive claim-evidence pairs across Wikipedia revisions.
- **Role**: Ground-truth contradiction detection training and evaluation data.
- **Preprocessing**: Trims whitespace, maps labels to standard identifiers, preserves revision metadata, and extracts contrastive pairs.

---

## Project Structure

```
├── 01_multinews_eda.py           # Multi-News EDA, preprocessing, and export
├── 02_wcep10_eda.py              # WCEP-10 EDA, preprocessing, and export
├── 03_vitaminc_eda.py            # VitaminC EDA, preprocessing, and export
├── 04_tfidf_retrieval.py         # TF-IDF sparse retrieval baseline
├── 05_dense_retrieval.py         # Dense retrieval (Sentence-BERT + FAISS) with TF-IDF comparison
├── 06_claim_extraction.py        # Stage 1: Atomic factual claim extractor
├── 07_nli_classifier.py          # Stage 2: Factual / DeBERTa NLI 3-way classifier
├── contextual_verification.py    # Stage 3: LLM Contextual Reasoner (conditions, scope, temporal)
├── conflict_resolution.py        # Stage 4: Authority, recency, and precedence resolution
├── contradiction_graph.py        # Stage 5: Contradiction Knowledge Graph & persistent memory
├── evaluation_benchmark.py       # Stage 6: 3-way architecture benchmark on VitaminC / news
├── run_contradiction_pipeline.py # Master unified pipeline (CLI, interactive, demo)
├── output/
│   ├── multinews/                # Cleaned sample CSV, stats, EDA visuals
│   ├── tfidf_index/              # TF-IDF index stats and cache
│   ├── dense_index/              # Dense vector embeddings and stats
│   ├── contradiction_graph/      # Persistent knowledge_graph.json
│   ├── pipeline_runs/            # Full JSON run traces and structured outputs
│   └── benchmark/                # 3-way benchmark evaluation results
├── PROJECT_DOCUMENTATION.md      # Comprehensive technical documentation
└── README.md
```

> **Note**: Cleaned `.csv`, `.jsonl`, `.npy`, and `.bin` files are generated locally in `output/` when the scripts are executed.

---

## Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install datasets pandas numpy matplotlib scikit-learn sentence-transformers faiss-cpu
   ```

2. **Run the End-to-End Contradiction Pipeline:**
   ```bash
   # One-shot CLI query over the Multi-News corpus:
   python run_contradiction_pipeline.py --query "hurricane death toll Florida" --top_k 3

   # Interactive Query REPL:
   python run_contradiction_pipeline.py --interactive

   # Run automated multi-perspective dispute demonstrations (Maria, Leave Policy, Vaccine Efficacy):
   python run_contradiction_pipeline.py --demo
   ```

3. **Run Individual Pipeline Stages:**
   ```bash
   # Stage 1: Atomic Claim Extraction
   python 06_claim_extraction.py

   # Stage 2: NLI Pairwise Classification
   python 07_nli_classifier.py

   # Stage 3: Contextual Verification Reasoner
   python contextual_verification.py

   # Stage 4: Conflict Resolution Layer
   python conflict_resolution.py

   # Stage 5: Contradiction Knowledge Graph
   python contradiction_graph.py

   # Stage 6: Comparative 3-Way Benchmark
   python evaluation_benchmark.py
   ```

---

## 3-Way Evaluation Benchmark Results

We benchmarked three architectural configurations on a curated test suite of factual contradictions, conditional exceptions, temporal revisions, and scope variations:

| Configuration | Precision | Recall | F1-Score | FPR (Contextual) | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NLI Alone (DeBERTa / Factual)** | 62.5% | 100.0% | 0.769 | 75.0% | ~1.9 ms |
| **Reasoner Alone (LLM Verification)** | 100.0% | 100.0% | 1.000 | 0.0% | ~1.8 ms |
| **Hybrid (NLI Gating + Reasoner)** | **100.0%** | **100.0%** | **1.000** | **0.0%** | **~1.6 ms** |

**Key Finding:** NLI alone flags conditional differences (e.g. 24 days general leave vs 12 days for first-year staff joining after July) as contradictions (75% False Positive Rate). The **Hybrid NLI + Reasoner pipeline** eliminates these false positives while preserving 100% recall on actual factual contradictions.
