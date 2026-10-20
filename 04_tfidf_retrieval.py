"""
TF-IDF Document Retrieval for Contradiction-Aware RAG
======================================================
Phase 2 -- Retrieval Layer

Builds a TF-IDF index over the Multi-News cleaned corpus (or falls back to
loading a small sample directly from Hugging Face if the CSV has not been
generated yet) and retrieves the top-k most relevant documents for any
free-text query.

Usage (interactive):
    python 04_tfidf_retrieval.py

Usage (CLI one-liner):
    python 04_tfidf_retrieval.py --query "hurricane death toll Florida" --top_k 5

Dependencies:
    pip install scikit-learn pandas numpy datasets
"""

import argparse
import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "multinews"
INDEX_DIR  = BASE_DIR / "output" / "tfidf_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV = OUTPUT_DIR / "multinews_cleaned_sample.csv"

# ---------------------------------------------------------------------------
# Compatibility shims (same as EDA scripts)
# ---------------------------------------------------------------------------
try:
    import huggingface_hub.utils._hf_uris as _hf_uris
    _orig_parse = _hf_uris._parse_repo_body
    def _compat_parse(location, type_, raw):
        if location.startswith("multi_news@"):
            location = "alexfabbri/" + location
        return _orig_parse(location, type_, raw=raw)
    _hf_uris._parse_repo_body = _compat_parse
except Exception:
    pass

try:
    import datasets.utils._dill, dill
    def _compat_batch(self, items, *args, **kwargs):
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        return dill.Pickler._batch_setitems(self, items, *args, **kwargs)
    datasets.utils._dill.Pickler._batch_setitems = _compat_batch
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def _clean_article(text: str) -> str:
    """Minimal text cleaning matching 01_multinews_eda.py."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    return text


def load_corpus(sample_size: int = 2000) -> pd.DataFrame:
    """
    Load the document corpus.

    Priority:
      1. Pre-generated CSV from 01_multinews_eda.py (if it exists).
      2. Fresh sample downloaded from Hugging Face.

    Returns a DataFrame with columns: doc_id, cluster_id, text, summary.
    """
    if CLEANED_CSV.exists():
        print(f"[INFO] Loading pre-cleaned corpus from: {CLEANED_CSV}")
        df = pd.read_csv(CLEANED_CSV)
        print(f"[INFO] Loaded {len(df):,} articles.")
        return df

    print("[INFO] Pre-cleaned CSV not found -- downloading raw files from HuggingFace Hub...")
    # multi_news has no Parquet; the repo stores raw text files:
    #   data/train.src.cleaned  -- one cluster of articles per line (sep: |||||)
    #   data/train.tgt          -- one summary per line
    # We download them directly via huggingface_hub (no loading-script needed).
    from huggingface_hub import hf_hub_download

    src_path = hf_hub_download(
        repo_id   = "alexfabbri/multi_news",
        filename  = "data/train.src.cleaned",
        repo_type = "dataset",
    )
    tgt_path = hf_hub_download(
        repo_id   = "alexfabbri/multi_news",
        filename  = "data/train.tgt",
        repo_type = "dataset",
    )
    print(f"[INFO] Downloaded raw files to cache.")

    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        src_lines = f.readlines()
    with open(tgt_path, "r", encoding="utf-8", errors="replace") as f:
        tgt_lines = f.readlines()

    assert len(src_lines) == len(tgt_lines), "Mismatch between src and tgt line counts"
    n = min(sample_size, len(src_lines))
    src_lines = src_lines[:n]
    tgt_lines = tgt_lines[:n]
    print(f"[INFO] Using {n:,} clusters from {len(src_lines):,} total train rows.")

    rows = []
    for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        articles = src_line.strip().split("|||||")
        summary  = tgt_line.strip()
        if summary.startswith(("-", "\u2013", "\u2014")):
            summary = summary[1:].strip()

        for art_idx, art_text in enumerate(articles):
            art_text = _clean_article(art_text)
            if not art_text or len(art_text.split()) < 10:
                continue
            rows.append({
                "doc_id":       f"mn_{idx:05d}_{art_idx:02d}",
                "cluster_id":   f"mn_cluster_{idx:05d}",
                "text":         art_text,
                "word_count":   len(art_text.split()),
                "summary":      summary,
                "content_hash": hashlib.md5(art_text.encode()).hexdigest()[:12],
            })

    df = pd.DataFrame(rows)
    print(f"[INFO] Built corpus with {len(df):,} articles from {n:,} clusters.")
    return df


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------
class TFIDFRetriever:
    """
    Builds and queries a TF-IDF index over a document corpus.

    Parameters
    ----------
    max_features : int
        Vocabulary size cap.
    ngram_range  : tuple
        n-gram range, e.g. (1, 2) for unigrams + bigrams.
    min_df       : int
        Minimum document frequency per term.
    """

    def __init__(
        self,
        max_features: int   = 50_000,
        ngram_range:  tuple = (1, 2),
        min_df:       int   = 2,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features  = max_features,
            ngram_range   = ngram_range,
            min_df        = min_df,
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
            stop_words    = "english",
        )
        self.tfidf_matrix = None
        self.corpus_df    = None

    def build(self, corpus_df: pd.DataFrame) -> None:
        """Fit the vectorizer and build the document-term matrix."""
        self.corpus_df = corpus_df.reset_index(drop=True)
        print(f"[INFO] Fitting TF-IDF on {len(corpus_df):,} documents ...")
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_df["text"])
        vocab_size = len(self.vectorizer.vocabulary_)
        print(
            f"[INFO] Matrix shape: {self.tfidf_matrix.shape} | "
            f"Vocab: {vocab_size:,} terms"
        )

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Retrieve top-k documents ranked by cosine similarity to the query.

        Returns DataFrame with:
            rank, doc_id, cluster_id, score, word_count, snippet, summary
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built. Call .build(corpus_df) first.")

        query_vec   = self.vectorizer.transform([query])
        scores      = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.corpus_df.iloc[idx]
            results.append({
                "rank":       rank,
                "doc_id":     row["doc_id"],
                "cluster_id": row.get("cluster_id", ""),
                "score":      round(float(scores[idx]), 6),
                "word_count": row.get("word_count", len(row["text"].split())),
                "snippet":    row["text"][:300].replace("\n", " ") + "...",
                "full_text":  row["text"],
                "summary":    str(row.get("summary", "")),
            })

        return pd.DataFrame(results)

    def save_index_stats(self, out_path: Path) -> None:
        """Persist index metadata to JSON."""
        stats = {
            "num_documents": int(self.tfidf_matrix.shape[0]),
            "vocab_size":    int(self.tfidf_matrix.shape[1]),
            "ngram_range":   list(self.vectorizer.ngram_range),
            "max_features":  self.vectorizer.max_features,
            "min_df":        self.vectorizer.min_df,
            "sublinear_tf":  self.vectorizer.sublinear_tf,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"[INFO] Index stats saved to: {out_path}")


# ---------------------------------------------------------------------------
# Display / Persistence Helpers
# ---------------------------------------------------------------------------
def _print_results(query: str, results_df: pd.DataFrame) -> None:
    sep = "-" * 72
    print(f"\n{'=' * 72}")
    print(f"  QUERY  : {query}")
    print(f"  TOP-{len(results_df)} RESULTS  (TF-IDF cosine similarity)")
    print(f"{'=' * 72}")

    for _, row in results_df.iterrows():
        print(f"\n  Rank #{row['rank']}  |  Score: {row['score']:.6f}")
        print(f"  Doc ID     : {row['doc_id']}")
        print(f"  Cluster    : {row['cluster_id']}")
        print(f"  Words      : {row['word_count']:,}")
        print(sep)
        print(f"  Snippet:")
        # Wrap snippet at 68 chars for readability
        snippet = row["snippet"]
        for i in range(0, min(len(snippet), 280), 68):
            print(f"    {snippet[i:i+68]}")
        if row["summary"]:
            summary_short = row["summary"][:200].replace("\n", " ")
            print(f"  Summary    : {summary_short} ...")
        print()

    print(f"{'=' * 72}\n")


def _save_results(query: str, results_df: pd.DataFrame) -> None:
    safe_query = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:40]
    out_path   = INDEX_DIR / f"results_{safe_query}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for _, row in results_df.iterrows():
            record = row.to_dict()
            record.pop("full_text", None)
            record["query"] = query
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Interactive Loop
# ---------------------------------------------------------------------------
def interactive_loop(retriever: TFIDFRetriever) -> None:
    print("\n[INTERACTIVE MODE]")
    print("  Type a query and press Enter to retrieve documents.")
    print("  Append  --top_k <n>  to change result count (default: 5).")
    print("  Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            print("[INFO] Goodbye.")
            break

        # Allow inline --top_k override
        top_k = 5
        if "--top_k" in raw:
            parts = raw.split("--top_k")
            raw   = parts[0].strip()
            try:
                top_k = int(parts[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

        results = retriever.retrieve(raw, top_k=top_k)

        if results.empty or results["score"].max() == 0:
            print("[WARN] No relevant documents found. Try different keywords.\n")
            continue

        _print_results(raw, results)
        _save_results(raw, results)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="TF-IDF document retrieval over the Multi-News corpus."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query string. Omit to launch interactive mode.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=2000,
        help="Articles to fetch from HF if CSV not found (default: 2000).",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=50_000,
        help="TF-IDF vocabulary size cap (default: 50000).",
    )
    args = parser.parse_args()

    # 1. Load corpus
    corpus_df = load_corpus(sample_size=args.sample_size)

    # 2. Build index
    retriever = TFIDFRetriever(max_features=args.max_features)
    retriever.build(corpus_df)
    retriever.save_index_stats(INDEX_DIR / "tfidf_index_stats.json")

    # 3. Retrieve
    if args.query:
        results = retriever.retrieve(args.query, top_k=args.top_k)
        _print_results(args.query, results)
        _save_results(args.query, results)
    else:
        interactive_loop(retriever)


if __name__ == "__main__":
    main()
