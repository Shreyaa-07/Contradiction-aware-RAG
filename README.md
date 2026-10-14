# Contradiction-Aware RAG

An exploratory data analysis (EDA), preprocessing, and dataset feasibility pipeline for building a **Contradiction-Aware Retrieval-Augmented Generation (RAG)** system.

---

## Overview

When standard RAG systems retrieve multiple documents across different news sources covering the same event, they often encounter conflicting or contradictory information (e.g., conflicting casualty counts, dates, or factual claims). This project explores multi-document corpora and fact-verification datasets to evaluate:
1. Suitable multi-document retrieval corpora (Multi-News, WCEP-10).
2. Labeled training/evaluation datasets for contradiction detection (VitaminC).

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
├── 01_multinews_eda.py      # Multi-News EDA, preprocessing, and export
├── 02_wcep10_eda.py         # WCEP-10 EDA, preprocessing, and export
├── 03_vitaminc_eda.py       # VitaminC EDA, preprocessing, and export
├── output/
│   ├── multinews/           # Visualizations (.png) and stats summary (.json)
│   ├── wcep10/              # Visualizations (.png) and stats summary (.json)
│   └── vitaminc/            # Visualizations (.png) and stats summary (.json)
├── .gitignore
└── README.md
```

> **Note**: Cleaned `.csv` and `.jsonl` files are generated locally in `output/` when the scripts are executed.

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shreyaa-07/Contradiction-aware-RAG.git
   cd Contradiction-aware-RAG
   ```

2. **Install dependencies:**
   ```bash
   pip install datasets pandas numpy matplotlib
   ```

3. **Run the EDA scripts:**
   ```bash
   python 01_multinews_eda.py
   python 02_wcep10_eda.py
   python 03_vitaminc_eda.py
   ```
