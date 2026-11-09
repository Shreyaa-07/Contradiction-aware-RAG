"""
Dense Retrieval for Contradiction-Aware RAG
============================================
Phase 2 -- Dense (Embedding-Based) Retrieval Layer

Encodes the Multi-News cleaned corpus with Sentence-BERT, builds a FAISS index,
and retrieves top-k documents via cosine similarity in embedding space.  Also
runs a head-to-head comparison against the TF-IDF baseline from 04_tfidf_retrieval.py.

Usage (interactive):
    python 05_dense_retrieval.py

Usage (CLI one-liner):
    python 05_dense_retrieval.py --query "hurricane death toll Florida" --top_k 5

Usage (compare against TF-IDF):
    python 05_dense_retrieval.py --compare

Dependencies:
    pip install sentence-transformers faiss-cpu pandas numpy datasets scikit-learn
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "multinews"
INDEX_DIR  = BASE_DIR / "output" / "dense_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_CSV = OUTPUT_DIR / "multinews_cleaned_sample.csv"
EMBEDDINGS_FILE = INDEX_DIR / "corpus_embeddings.npy"
FAISS_INDEX_FILE = INDEX_DIR / "faiss_index.bin"
INDEX_META_FILE = INDEX_DIR / "dense_index_stats.json"

# ---------------------------------------------------------------------------
# HF compatibility shims (same as other scripts)
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
# Corpus loading (shared with 04_tfidf_retrieval.py)
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
        print(f"Loading corpus from {CLEANED_CSV}")
        df = pd.read_csv(CLEANED_CSV)
        return df

    print("CSV not found. Downloading from Hugging Face...")
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

    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        src_lines = f.readlines()
    with open(tgt_path, "r", encoding="utf-8", errors="replace") as f:
        tgt_lines = f.readlines()

    assert len(src_lines) == len(tgt_lines), "Mismatch between src and tgt line counts"
    n = min(sample_size, len(src_lines))
    src_lines = src_lines[:n]
    tgt_lines = tgt_lines[:n]

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
    return df


# ---------------------------------------------------------------------------
# Dense Retriever
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Builds and queries a FAISS index over Sentence-BERT embeddings.

    Parameters
    ----------
    model_name : str
        Sentence-BERT model from HuggingFace. Default is 'all-MiniLM-L6-v2'
        (384-dim, fast, good quality).
    batch_size : int
        Encoding batch size. Lower if you run out of memory.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer
        import faiss

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.corpus_df = None
        self.embeddings = None

    def build(self, corpus_df: pd.DataFrame, use_cache: bool = True) -> None:
        """
        Encode documents and build the FAISS index.

        If cached embeddings exist on disk and use_cache=True, loads from disk
        instead of re-encoding (saves significant time on repeat runs).
        """
        import faiss

        self.corpus_df = corpus_df.reset_index(drop=True)
        texts = corpus_df["text"].tolist()

        # Check cache
        if use_cache and EMBEDDINGS_FILE.exists():
            print(f"Loading cached embeddings from {EMBEDDINGS_FILE}")
            self.embeddings = np.load(str(EMBEDDINGS_FILE))
            if self.embeddings.shape[0] != len(texts):
                print(f"Cache mismatch ({self.embeddings.shape[0]} vs {len(texts)} docs). Re-encoding...")
                self.embeddings = None

        if self.embeddings is None:
            print(f"Encoding {len(texts):,} documents (batch_size={self.batch_size})...")
            start_time = time.time()
            self.embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )
            elapsed = time.time() - start_time
            print(f"Encoding complete in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/sec)")

            # Cache to disk
            np.save(str(EMBEDDINGS_FILE), self.embeddings)
            print(f"Saved embeddings to {EMBEDDINGS_FILE}")

        # Build FAISS index (Inner Product = cosine similarity on L2-normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))

        # Save FAISS index
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        print(f"FAISS index built: {self.index.ntotal:,} vectors, {self.embedding_dim}D")

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Retrieve top-k documents ranked by cosine similarity to the query.

        Returns DataFrame with:
            rank, doc_id, cluster_id, score, word_count, snippet, summary
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call .build(corpus_df) first.")

        # Encode query
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search FAISS
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
            row = self.corpus_df.iloc[idx]
            results.append({
                "rank":       rank,
                "doc_id":     row["doc_id"],
                "cluster_id": row.get("cluster_id", ""),
                "score":      round(float(score), 6),
                "word_count": row.get("word_count", len(row["text"].split())),
                "snippet":    row["text"][:300].replace("\n", " ") + "...",
                "full_text":  row["text"],
                "summary":    str(row.get("summary", "")),
            })

        return pd.DataFrame(results)

    def save_index_stats(self, out_path: Path) -> None:
        """Persist index metadata to JSON."""
        stats = {
            "model_name":     self.model_name,
            "embedding_dim":  int(self.embedding_dim),
            "num_documents":  int(self.index.ntotal),
            "index_type":     "FlatIP (exact cosine search)",
            "batch_size":     self.batch_size,
            "embeddings_cached": str(EMBEDDINGS_FILE),
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)
        print(f"Saved index stats to {out_path}")


# ---------------------------------------------------------------------------
# TF-IDF Retriever (lightweight re-import for comparison)
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    """Minimal TF-IDF retriever for side-by-side comparison."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.vectorizer = TfidfVectorizer(
            max_features  = 50_000,
            ngram_range   = (1, 2),
            min_df        = 2,
            sublinear_tf  = True,
            strip_accents = "unicode",
            analyzer      = "word",
            token_pattern = r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
            stop_words    = "english",
        )
        self.tfidf_matrix = None
        self.corpus_df = None
        self._cosine_similarity = cosine_similarity

    def build(self, corpus_df: pd.DataFrame) -> None:
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_df["text"])

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec   = self.vectorizer.transform([query])
        scores      = self._cosine_similarity(query_vec, self.tfidf_matrix).flatten()
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
                "summary":    str(row.get("summary", "")),
            })

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Comparison: Dense vs TF-IDF
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    "hurricane death toll Florida",
    "election results controversy vote count",
    "climate change global warming temperature",
    "police shooting unarmed victim",
    "earthquake casualties damage estimate",
    "trade war tariffs China United States",
    "COVID vaccine side effects safety",
    "oil prices stock market crash",
    "school shooting gun control",
    "immigration border policy refugees",
]


def _safe_str(val: object) -> str:
    """Sanitize strings for Windows console environments."""
    return str(val).encode("ascii", "replace").decode("ascii")


def run_comparison(dense_retriever: DenseRetriever, corpus_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Run both retrievers on benchmark queries and compare results.

    Computes:
      - Overlap@k: how many documents appear in both result sets
      - Cluster diversity: how many unique clusters each retriever returns
      - Rank correlation: Spearman correlation of overlapping doc ranks
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Dense Retrieval (Sentence-BERT) vs TF-IDF")
    print("=" * 80)

    # Build TF-IDF baseline
    print("\nBuilding TF-IDF baseline index...")
    tfidf = TFIDFRetriever()
    tfidf.build(corpus_df)

    comparison_rows = []

    for query in BENCHMARK_QUERIES:
        dense_results = dense_retriever.retrieve(query, top_k=top_k)
        tfidf_results = tfidf.retrieve(query, top_k=top_k)

        dense_docs = set(dense_results["doc_id"].tolist())
        tfidf_docs = set(tfidf_results["doc_id"].tolist())
        overlap = dense_docs & tfidf_docs

        dense_clusters = dense_results["cluster_id"].nunique()
        tfidf_clusters = tfidf_results["cluster_id"].nunique()

        dense_avg_score = dense_results["score"].mean()
        tfidf_avg_score = tfidf_results["score"].mean()

        comparison_rows.append({
            "query":             query,
            "overlap":           len(overlap),
            "dense_clusters":    dense_clusters,
            "tfidf_clusters":    tfidf_clusters,
            "dense_avg_score":   round(dense_avg_score, 4),
            "tfidf_avg_score":   round(tfidf_avg_score, 4),
        })

        print(f"\nQuery: \"{query}\"")
        print(f"  Overlap@{top_k}: {len(overlap)}/{top_k} docs in common")
        print(f"  Dense clusters: {dense_clusters} | TF-IDF clusters: {tfidf_clusters}")

        # Show side-by-side top-1
        if not dense_results.empty and not tfidf_results.empty:
            d1 = dense_results.iloc[0]
            t1 = tfidf_results.iloc[0]
            print(f"  Dense  #1: [{d1['doc_id']}] score={d1['score']:.4f} | {_safe_str(d1['snippet'][:80])}")
            print(f"  TF-IDF #1: [{t1['doc_id']}] score={t1['score']:.4f} | {_safe_str(t1['snippet'][:80])}")

    comp_df = pd.DataFrame(comparison_rows)

    # Summary statistics
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  Average Overlap@{top_k}: {comp_df['overlap'].mean():.1f}/{top_k}")
    print(f"  Dense avg unique clusters: {comp_df['dense_clusters'].mean():.1f}")
    print(f"  TF-IDF avg unique clusters: {comp_df['tfidf_clusters'].mean():.1f}")
    print(f"  Queries where dense found MORE clusters: "
          f"{(comp_df['dense_clusters'] > comp_df['tfidf_clusters']).sum()}/{len(comp_df)}")

    # Save comparison results
    comp_path = INDEX_DIR / "tfidf_vs_dense_comparison.json"
    comp_df.to_json(comp_path, orient="records", indent=2)
    print(f"\nSaved comparison to {comp_path}")

    return comp_df


# ---------------------------------------------------------------------------
# Display and persistence helpers
# ---------------------------------------------------------------------------

def _print_results(query: str, results_df: pd.DataFrame) -> None:
    print(f"\nQUERY: {query}")
    print(f"TOP-{len(results_df)} RESULTS (Dense / Sentence-BERT cosine similarity)\n")

    for _, row in results_df.iterrows():
        print(f"  Rank #{row['rank']}  |  Score: {row['score']:.6f}")
        print(f"  Doc ID     : {row['doc_id']}")
        print(f"  Cluster    : {row['cluster_id']}")
        print(f"  Words      : {row['word_count']:,}")
        snippet = _safe_str(row["snippet"][:200])
        print(f"  Snippet    : {snippet}")
        if row["summary"]:
            summary_short = _safe_str(row["summary"][:200].replace("\n", " "))
            print(f"  Summary    : {summary_short}")
        print()


def _save_results(query: str, results_df: pd.DataFrame) -> None:
    safe_query = re.sub(r"[^a-zA-Z0-9]+", "_", query)[:40]
    out_path   = INDEX_DIR / f"dense_results_{safe_query}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for _, row in results_df.iterrows():
            record = row.to_dict()
            record.pop("full_text", None)
            record["query"] = query
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def interactive_loop(retriever: DenseRetriever) -> None:
    print("\nDense Retrieval interactive mode. Type a query or 'quit' to exit.")
    print("Append --top_k N to change result count (default 5).\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue
        if raw.lower() in {"exit", "quit", "q"}:
            break

        top_k = 5
        if "--top_k" in raw:
            parts = raw.split("--top_k")
            raw   = parts[0].strip()
            try:
                top_k = int(parts[1].strip().split()[0])
            except (IndexError, ValueError):
                pass

        results = retriever.retrieve(raw, top_k=top_k)

        if results.empty or results["score"].max() <= 0:
            print("  No results found.\n")
            continue

        _print_results(raw, results)
        _save_results(raw, results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dense document retrieval (Sentence-BERT + FAISS) over the Multi-News corpus."
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
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-BERT model to use (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Encoding batch size (default: 64). Lower if out of memory.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run side-by-side comparison against TF-IDF baseline.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Force re-encoding (ignore cached embeddings).",
    )
    args = parser.parse_args()

    # Load corpus
    corpus_df = load_corpus(sample_size=args.sample_size)
    print(f"Corpus loaded: {len(corpus_df):,} documents, "
          f"{corpus_df['cluster_id'].nunique():,} clusters")

    # Build dense retriever
    retriever = DenseRetriever(model_name=args.model, batch_size=args.batch_size)
    retriever.build(corpus_df, use_cache=not args.no_cache)
    retriever.save_index_stats(INDEX_META_FILE)

    if args.compare:
        run_comparison(retriever, corpus_df, top_k=args.top_k)
    elif args.query:
        results = retriever.retrieve(args.query, top_k=args.top_k)
        _print_results(args.query, results)
        _save_results(args.query, results)
    else:
        interactive_loop(retriever)


if __name__ == "__main__":
    main()

