# Contradiction-Aware RAG: Comprehensive Project Documentation & Analysis

**Course:** Information Retrieval and Extraction (IRE)  
**Project:** Contradiction-Aware Retrieval-Augmented Generation (RAG) System  
**Repository:** [Shreyaa-07/Contradiction-aware-RAG](https://github.com/Shreyaa-07/Contradiction-aware-RAG.git)  
**Author:** Shreya  

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [The Core Problem: Why Standard RAG Fails](#2-the-core-problem-why-standard-rag-fails)
3. [System Architecture & Data Requirements](#3-system-architecture--data-requirements)
4. [Dataset Deep Dive & EDA Findings](#4-dataset-deep-dive--eda-findings)
   - [4.1 Multi-News Dataset (`01_multinews_eda.py`)](#41-multi-news-dataset-01_multinews_edapy)
   - [4.2 WCEP-10 Dataset (`02_wcep10_eda.py`)](#42-wcep-10-dataset-02_wcep10_edapy)
   - [4.3 VitaminC Dataset (`03_vitaminc_eda.py`)](#43-vitaminc-dataset-03_vitaminc_edapy)
5. [Comparative Evaluation & Dataset Decision Matrix](#5-comparative-evaluation--dataset-decision-matrix)
6. [Why We Chose These Specific Datasets](#6-why-we-chose-these-specific-datasets)
7. [What Else Could We Have Done? (Alternative Paths Considered)](#7-what-else-could-we-have-done-alternative-paths-considered)
   - [7.1 Alternative Document Retrieval Corpora](#71-alternative-document-retrieval-corpora)
   - [7.2 Alternative Contradiction / NLI Datasets](#72-alternative-contradiction--nli-datasets)
   - [7.3 Alternative System Architectures](#73-alternative-system-architectures)
8. [Data Preprocessing & Engineering Decisions](#8-data-preprocessing--engineering-decisions)
9. [End-to-End Roadmap: What We Have Done vs. Next Steps](#9-end-to-end-roadmap-what-we-have-done-vs-next-steps)
10. [Engineering Challenges Encountered & Resolutions](#10-engineering-challenges-encountered--resolutions)

---

## 1. Executive Summary & Problem Statement

This project addresses a critical limitation in modern **Retrieval-Augmented Generation (RAG)** pipelines: **handling contradictory and conflicting information across multiple retrieved sources**.

In multi-source information retrieval, particularly in news reporting, different reputable sources frequently report conflicting factual claims about the same event (e.g., conflicting death tolls, disputed timelines, varying monetary estimates, or differing official statements). 

Standard RAG architectures assume that retrieved context is either relevant and truthful or irrelevant. When presented with mutually exclusive facts, standard LLMs either **hallucinate**, **arbitrarily pick one source**, or **blend contradictory numbers together** without alerting the user.

Our objective is to design and build a **Contradiction-Aware RAG system** that:
1. Retrieves multiple source documents relevant to a user query.
2. Identifies and isolates conflicting factual claims across the retrieved passages.
3. Synthesizes an informed, balanced answer that **explicitly surfaces the contradictions and explains the discrepancies**, citing the respective sources.

---

## 2. The Core Problem: Why Standard RAG Fails

```
               [ User Query: "What was the casualty count in the incident?" ]
                                             │
                                             ▼
                                    [ Vector Retrieval ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
         [ Source A: "20 people died" ]               [ Source B: "35 people died" ]
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                                  [ Standard LLM Prompt ]
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
             [ Failure Mode 1: Hallucination ]      [ Failure Mode 2: Silent Selection ]
           "Around 27 people died in the incident"    "20 people died" (ignores Source B)
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             ▼
                              ❌ Misleading / Inaccurate
```

### Common Failure Modes in Standard RAG:

1. **Averaging / Compromise Hallucination:** The LLM attempts to reconcile "$10 million" and "$50 million" by stating "tens of millions" or hallucinating a third figure.
2. **First-Token / Position Bias:** The model arbitrarily trusts whichever document happens to be placed first in the prompt context.
3. **Majority Fallacy:** If 3 syndicated wire copies repeat an unverified early report of "5 casualties" while 1 later investigative piece reports "25 casualties", standard RAG outputs "5 casualties" simply because of token frequency.
4. **Failure to Signal Uncertainty:** The user is left unaware that the factual basis of the answer is actively disputed.

---

## 3. System Architecture & Data Requirements

To solve this problem, the system requires two fundamentally distinct types of data:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA REQUIREMENTS                                 │
├─────────────────────────────────────────┬────────────────────────────────────────┤
│     1. DOCUMENT RETRIEVAL CORPUS        │    2. CONTRADICTION CLASSIFIER DATA    │
├─────────────────────────────────────────┼────────────────────────────────────────┤
│ • Must contain clusters of articles     │ • Must contain labeled pairs of claims │
│   covering the EXACT SAME event.        │   and evidence.                        │
│ • Natural, realistic news text.         │ • Ground-truth labels:                 │
│ • Rich in cross-source claims.          │   SUPPORTS, REFUTES, NOT ENOUGH INFO.  │
│ • Where contradictions NATURALLY occur. │ • Used to train/evaluate the NLI model.│
│                                         │                                        │
│ Candidate Datasets:                     │ Candidate Dataset:                     │
│ ➜ Multi-News                            │ ➜ VitaminC                             │
│ ➜ WCEP-10                               │                                        │
└─────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 4. Dataset Deep Dive & EDA Findings

### 4.1 Multi-News Dataset (`01_multinews_eda.py`)

* **Reference:** Fabbri et al. (2019), *Multi-News: A Large-Scale Multi-Document Summarization Dataset and Abstractive Hierarchical Model*, ACL 2019 ([arXiv:1906.01749](https://arxiv.org/abs/1906.01749)).
* **Source:** Hugging Face `alexfabbri/multi_news`
* **Structure:** Each instance contains a `document` string (multiple news articles concatenated by the token ` ||||| `) and an editor-written abstractive `summary` from Newser.com.

#### Quantitative EDA Results (from 2,000-Cluster Sample):

| Metric | Measured Value | Significance |
| :--- | :--- | :--- |
| **Total Training Clusters** | 44,972 clusters | Large-scale corpus for indexing |
| **Sample Clusters Evaluated** | 2,000 clusters | Statistically significant EDA sample |
| **Individual Articles Extracted** | 5,748 articles | ~2.88 articles per cluster |
| **Clusters with `\|\|\|\|\|` delimiter** | 1,979 (99.0%) | 1% are single-article clusters |
| **Articles per Cluster (Mean / Median / Max)** | 2.89 / 2.0 / 10 | Diverse multi-source coverage |
| **Article Word Count (Mean / Median)** | 638.7 / 461.0 words | Sufficient depth for chunking |
| **Cluster Total Word Count (Mean / Median)** | 1,849.6 / 1,390.5 words | Rich contextual breadth |
| **Summary Word Count (Mean / Median)** | 215.8 / 217.0 words | Multi-sentence synthesis |
| **Articles < 50 Words (Junk/Boilerplate)** | 421 articles (7.3%) | Filtered out during cleaning |
| **Exact Duplicate Articles by MD5** | 342 articles | Redundant syndicated articles |
| **Cleaned Articles Retained** | 5,748 articles | High yield (>99%) |

#### Preprocessing Performed:
1. **Delimiter Splitting:** Parsed the raw concatenated `document` text using the regex delimiter `\|\|\|\|\|`.
2. **Whitespace Normalization:** Collapsed runs of 3+ newlines into paragraph breaks (`\n\n`) and trimmed trailing whitespace.
3. **Editorial Formatting Removal:** Stripped leading dashes (`–`, `—`, `- `) typical of Newser editor conventions.
4. **Quality Filtering:** Discarded fragments under 10 words (boilerplate headers, captions, cookies warnings).
5. **Deterministic ID Generation:** Assigned persistent keys:
   - Cluster ID: `mn_cluster_00000`
   - Document ID: `mn_00000_01`
   - Content MD5 Hash: 12-character hash for instant deduplication.

---

### 4.2 WCEP-10 Dataset (`02_wcep10_eda.py`)

* **Reference:** Gholipour Ghalandari et al. (2020), *Large-Scale Multi-Document Summarization of News using Wikipedia Current Events Portal*, ACL 2020 ([arXiv:2005.10070](https://arxiv.org/abs/2005.10070)).
* **Source:** Hugging Face `ccdv/WCEP-10`
* **Structure:** Each record links to a Wikipedia Current Events Portal entry. Up to 10 articles are grouped per event, delimited by `</s>`, with a human-written Wikipedia summary.

#### Quantitative EDA Results (Full Training Split):

| Metric | Measured Value | Significance |
| :--- | :--- | :--- |
| **Total Training Clusters** | 8,158 clusters | Highly curated event-driven corpus |
| **Individual Articles Extracted** | 73,325 articles | High article density per event |
| **Delimiter Detected** | `</s>` | SentencePiece boundary token |
| **Articles per Cluster (Mean / Median / Max)** | 8.99 / 10.0 / 10 | High redundancy and multi-angle coverage |
| **Article Word Count (Mean / Median)** | 420.3 / 337.0 words | Focused news reports |
| **Cluster Total Word Count (Mean / Median)** | 3,770.0 / 3,536.5 words | Very high information density per event |
| **Summary Word Count (Mean / Median)** | 27.9 / 25.0 words | Concise Wikipedia-style factoid summary |
| **Articles < 50 Words** | 2,954 articles (4.0%) | Short stubs/snippets |
| **Exact Duplicate Articles by MD5** | 1,950 articles | Cross-posted wire stories (AP, Reuters) |
| **Cleaned Articles Retained** | 73,231 articles | Extremely clean extraction yield |

#### Preprocessing Performed:
1. **Token Parsing:** Extracted articles using `</s>` as the delimiter.
2. **Text Sanitation:** Normalized spacing and removed carriage returns.
3. **Filtering:** Enforced a 10-word minimum threshold to eliminate broken fragments.
4. **Standardized Schema:** Aligned output format with Multi-News (`doc_id`, `cluster_id`, `text`, `word_count`, `summary`, `content_hash`).

---

### 4.3 VitaminC Dataset (`03_vitaminc_eda.py`)

* **Reference:** Schuster, Fisch & Barzilay (2021), *Get Your Vitamin C! Robust Fact Verification with Contrastive Evidence*, NAACL 2021 ([arXiv:2104.10850](https://arxiv.org/abs/2104.10850)).
* **Source:** Hugging Face `tals/vitaminc`
* **Structure:** Fact verification pairs extracted from Wikipedia revision histories. Contains contrastive revisions where a subtle factual update changes the verification status of a claim from `SUPPORTS` to `REFUTES`.

#### Quantitative EDA Results (from 5,000-Pair Sample):

| Metric | Measured Value | Significance |
| :--- | :--- | :--- |
| **Total Training Pairs** | 370,653 pairs | Vast dataset for NLI fine-tuning |
| **Validation / Test Pairs** | 63,054 / 55,197 | Standardized benchmark splits |
| **Label Balance: SUPPORTS** | 2,475 pairs (49.5%) | Balanced positive class |
| **Label Balance: REFUTES** | 1,769 pairs (35.4%) | Strong contradiction signal |
| **Label Balance: NOT ENOUGH INFO** | 756 pairs (15.1%) | Realistic neutrality/insufficiency class |
| **Claim Word Count (Mean / Median)** | 12.7 / 11.0 words | Single proposition claims |
| **Evidence Word Count (Mean / Median)** | 29.7 / 26.0 words | Single-to-two sentence Wikipedia excerpts |
| **Unique Case IDs in Sample** | 4,916 case IDs | Contrastive groupings |
| **Contrastive Multi-Label Cases** | 52 groups in sample (1.1%) | Same claim with opposing labels |
| **Unique Wikipedia Pages** | 3,536 pages | High domain diversity |

#### Real Contrastive Example Discovered:
* **Claim:** *"Based on more than 15 reviewers, Unfinished Business scored less 7% with an approval rating."*
* **Evidence Revision 1 (SUPPORTS):** *"The review aggregator website Rotten Tomatoes reported a 6% approval rating..."* ➔ **Verdict: SUPPORTS**
* **Evidence Revision 2 (REFUTES):** *"The review aggregator website Rotten Tomatoes reported a 9% approval rating..."* ➔ **Verdict: NOT ENOUGH INFO / REFUTES**

This demonstrates the exact type of subtle numerical contradiction our RAG system must catch.

---

## 5. Comparative Evaluation & Dataset Decision Matrix

| Dimension | Multi-News | WCEP-10 | VitaminC |
| :--- | :--- | :--- | :--- |
| **Format** | Multi-article news clusters | Multi-article event clusters | Claim–Evidence sentence pairs |
| **Observation Unit** | Cluster of full articles | Cluster of full articles | Single sentence pair |
| **Total Records** | 44,972 clusters (~130k articles) | 8,158 clusters (73,231 articles) | 370,653 pairs |
| **Articles per Topic** | 2–10 articles (avg 2.89) | 1–10 articles (avg 8.99) | N/A (1 sentence pair) |
| **Average Word Count** | ~638 words/article | ~420 words/article | ~30 words/evidence |
| **Contradiction Labels?** | ❌ No (requires discovery) | ❌ No (requires discovery) | ✅ Yes (`SUPPORTS`/`REFUTES`) |
| **Source/Date Metadata** | ⚠️ Limited in HF version | ⚠️ Limited in HF version | ✅ Detailed (Wikipedia Revision ID) |
| **Project Role** | **Primary Retrieval Corpus** | **Alternative / Benchmark Corpus** | **Contradiction Classifier Data** |

---

## 6. Why We Chose These Specific Datasets

1. **Multi-News as Primary Retrieval Corpus:**
   - Multi-News mimics the real-world RAG setting: full-length journalistic articles from diverse outlets (CNN, BBC, local news, blogs) reporting on the same event.
   - The natural divergence in reporting guarantees cross-source claim variation (different casualty estimates, unnamed sources, changing facts).

2. **WCEP-10 as Alternative / Development Corpus:**
   - With an average of ~9 articles per cluster, WCEP-10 has higher topical concentration than Multi-News.
   - Its Wikipedia summaries serve as an objective baseline for automated evaluation.
   - At 8,158 clusters, it is compact enough to vectorize completely in local vector databases during rapid prototyping.

3. **VitaminC as the Ground-Truth Evaluator:**
   - Standard NLI datasets (like SNLI or MNLI) test general linguistic entailment (e.g., *"A dog running" entails "An animal is moving"*).
   - In contrast, VitaminC tests **factual revisions** (e.g., changing numbers, dates, outcomes). This matches the exact type of contradiction found in news RAG.
   - It provides ground truth for benchmarking cross-encoder classifiers before deploying them on unlabeled Multi-News clusters.

---

## 7. What Else Could We Have Done? (Alternative Paths Considered)

### 7.1 Alternative Document Retrieval Corpora

| Candidate | Why Considered | Why Deferred or Rejected |
| :--- | :--- | :--- |
| **NewsRoom Dataset** | 1.3M articles with varied summary styles. | Single-document only (1 article ➔ 1 summary). Cannot be used for multi-source conflict retrieval without expensive clustering. |
| **DUC / TAC Multi-Doc Summarization** | Gold standard academic benchmark for multi-document summarization. | Extremely small scale (~50 clusters total); restricted licensing; outdated topics (late 1990s/early 2000s). |
| **Live News Scraping (NewsAPI / GNews)** | 100% current, live contradictions. | Non-reproducible; paywalled; rate-limited; violates course reproducibility requirements. |
| **WikiNews Revisions** | Authentic historical edits with timestamps. | Text volume is sparse; many articles are very short stubs; parsing revision diffs introduces immense noise. |

---

### 7.2 Alternative Contradiction / NLI Datasets

| Candidate | Description | Why VitaminC Was Preferred |
| :--- | :--- | :--- |
| **SNLI / MNLI** | Sentence-pair NLI based on image captions and transcribed speech. | Focuses on commonsense logic, lexical ambiguity, and syntax, rather than **factual knowledge conflicts**. |
| **FEVER** | Fact extraction and verification dataset based on Wikipedia. | Earlier predecessor to VitaminC. Lacks contrastive pairs; many synthetic refutations are grammatically artificial or trivial (e.g., adding "never" or "did not"). |
| **ANLI (Adversarial NLI)** | Multi-round crowd-sourced adversarial examples. | Hard linguistic puzzles, but lacks the specific temporal/news revision dynamics of VitaminC. |
| **WANLI** | Worker-and-AI collaborative dataset. | High quality, but does not provide direct Wikipedia factual revision lineage like VitaminC. |

---

### 7.3 Alternative System Architectures

| Approach | Description | Trade-offs & Analysis |
| :--- | :--- | :--- |
| **Approach A: Pure LLM Prompting (Zero-Shot)** | Send all retrieved articles to GPT-4/Claude with prompt: *"Point out any contradictions."* | **Pros:** Trivial to implement.<br>**Cons:** Expensive, high latency, context-window limits, highly prone to missing subtle numerical contradictions, cannot evaluate precision/recall objectively. |
| **Approach B: OpenIE / Triplet Extraction + Symbolic Logic** | Extract `(subject, predicate, object)` triplets (e.g. via spaCy/Stanford CoreNLP) and check for conflicting objects. | **Pros:** Explainable symbolic reasoning.<br>**Cons:** Brittle; fails on complex coreference, paraphrasing, and implicit claims in free-form journalistic prose. |
| **Approach C: Hybrid Modular Pipeline (Selected Architecture)** | 1. Vector Retrieval (Chroma/FAISS) ➔ 2. Sentence-level Claim Extractor ➔ 3. NLI Cross-Encoder (fine-tuned on VitaminC) ➔ 4. Contradiction-Aware Synthesis LLM. | **Pros:** Combines scalable dense retrieval, verifiable contradiction scoring (precision/recall metrics), and fluent natural-language synthesis. |

---

## 8. Data Preprocessing & Engineering Decisions

```
[ Raw Concatenated String ]
           │
           ▼
[ Delimiter Split (||||| or </s>) ]
           │
           ▼
[ Text Cleaning: Collapse Whitespace, Strip Leading Dashes ]
           │
           ▼
[ Length Threshold Filter (Drop articles < 10 words) ]
           │
           ▼
[ Deterministic ID & MD5 Hash Computation ]
           │
           ▼
[ Cleaned Schema Export (.csv, .jsonl, _stats.json) ]
```

### Key Engineering Choices:

1. **Deterministic Hashing:** Used MD5 content hashing on normalized text. This immediately flagged 342 identical duplicate articles in Multi-News and 1,950 in WCEP-10 caused by syndicated wire republication (e.g., Reuters articles reprinted across multiple newspapers).
2. **Short Fragment Pruning:** Removing articles under 10 words purged copyright notices, navigation crumbs, and broken web stubs without losing analytical content.
3. **Cross-Platform Console Encoding:** Standardized all print outputs to avoid non-ASCII characters and decorative Unicode symbols that crash Windows consoles running under `cp1252` encoding.
4. **Git LFS Avoidance via Targeted `.gitignore`:** Excluded 450+ MB of `.csv` and `.jsonl` artifacts from Git tracking, preventing GitHub's 100 MB hard reject while tracking visualizations and metadata statistics.

---

## 9. End-to-End Roadmap: What We Have Done vs. Next Steps

```
  PHASE 1: EDA & Preprocessing      [ COMPLETED ✅ ]
  ├── 01_multinews_eda.py           ➔ output/multinews/
  ├── 02_wcep10_eda.py              ➔ output/wcep10/
  ├── 03_vitaminc_eda.py            ➔ output/vitaminc/
  └── GitHub Repo Configured        ➔ Shreyaa-07/Contradiction-aware-RAG
           │
           ▼
  PHASE 2: Chunking & Indexing      [ NEXT STEP ⏳ ]
  ├── Semantic chunking (200-300 words per chunk)
  ├── Embedding model selection (e.g. all-MiniLM-L6-v2 or BGE-small)
  └── Vector store setup (ChromaDB / FAISS)
           │
           ▼
  PHASE 3: Contradiction Detection  [ UPCOMING ⏳ ]
  ├── Cross-encoder NLI model (RoBERTa / DeBERTa-v3)
  ├── Calibrate thresholds on VitaminC test split
  └── Pairwise retrieved passage contradiction scoring
           │
           ▼
  PHASE 4: Contradiction-Aware RAG  [ FINAL STAGE ⏳ ]
  ├── Conflict-structured prompt engineering
  ├── Source-attributed synthesis
  └── Evaluation (Faithfulness, Contradiction Recall, ROUGE)
```

---

## 10. Engineering Challenges Encountered & Resolutions

### 1. Windows Console `UnicodeEncodeError`
* **Issue:** Scripts crashed on Windows shells (`cp1252` encoding) when printing Unicode emojis (`📋`, `✅`, `⚠️`, `📊`).
* **Resolution:** Stripped decorative emojis and replaced them with standard ASCII text markers, ensuring portable execution across Linux, macOS, and Windows.

### 2. Hugging Face Hub URI Resolution in Newer Versions
* **Issue:** In `huggingface-hub >= 2.0`, canonical datasets without an explicit user namespace (like `multi_news`) caused URI parsing errors (`Repository id must be 'namespace/name'`).
* **Resolution:** Built a lightweight runtime compatibility shim in Python that intercepts legacy canonical URIs and correctly aliases `multi_news` to `alexfabbri/multi_news`.

### 3. Python 3.14 Pickle Serialization Compatibility
* **Issue:** Python 3.14 modified internal pickle signatures in `_batch_setitems()`, causing incompatibilities with `dill` in older `datasets` releases.
* **Resolution:** Implemented an argument-tolerant monkeypatch for `_batch_setitems` in the script headers, ensuring the scripts run across Python 3.11, 3.12, 3.13, and 3.14.

### 4. GitHub File Size Ceiling (100 MB Limit)
* **Issue:** WCEP-10 cleaned files exceeded 200 MB, which would cause `git push` rejection.
* **Resolution:** Configured `.gitignore` to track reproducible scripts, lightweight stats summaries, and visualization images while ignoring generated bulk tabular files.
