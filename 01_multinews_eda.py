"""
EDA and preprocessing for the Multi-News dataset.
Paper: Fabbri et al. (2019) — arXiv:1906.01749
"""

import hashlib
import json
import os
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "multinews")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compatibility helper for environments with Python 3.14 / newer HF hub
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

# Load Multi-News dataset
ds = load_dataset("alexfabbri/multi_news")
for split_name in ds:
    print(f"{split_name}: {len(ds[split_name]):,} records")

# Sample 2,000 records from train for EDA
SAMPLE_SIZE = 2000
sample_indices = list(range(min(SAMPLE_SIZE, len(ds["train"]))))
df = ds["train"].select(sample_indices).to_pandas()
print(f"Sample shape: {df.shape}")

# Data quality checks
missing = df.isnull().sum()
empty_strings = (df.apply(lambda col: col.astype(str).str.strip() == "")).sum()
print("Missing values per column:\n", missing)
print("Empty string records per column:\n", empty_strings)

full_dupes = df.duplicated().sum()
doc_dupes = df["document"].duplicated().sum()
summary_dupes = df["summary"].duplicated().sum()
print(f"Duplicates - Full rows: {full_dupes}, Documents: {doc_dupes}, Summaries: {summary_dupes}")

# Multi-News separates articles within each cluster by ' ||||| '
has_delimiter = df["document"].str.contains(r"\|\|\|\|\|", regex=True).sum()
print(f"Clusters containing ' ||||| ' delimiter: {has_delimiter}/{len(df)} ({has_delimiter/len(df)*100:.1f}%)")

df["article_count"] = df["document"].apply(lambda x: len(x.split("|||||")))
print("Articles per cluster summary:")
print(df["article_count"].describe())

# Text length analysis
df["doc_char_len"] = df["document"].str.len()
df["doc_word_count"] = df["document"].str.split().str.len()
df["summary_char_len"] = df["summary"].str.len()
df["summary_word_count"] = df["summary"].str.split().str.len()

print("Cluster and summary word count statistics:")
print(df[["doc_word_count", "summary_word_count"]].describe())

# Individual article extraction
all_articles = []
for idx, row in df.iterrows():
    articles = row["document"].split("|||||")
    for i, art in enumerate(articles):
        art_text = art.strip()
        if art_text:
            all_articles.append({
                "cluster_idx": idx,
                "article_idx": i,
                "text": art_text,
                "char_len": len(art_text),
                "word_count": len(art_text.split()),
            })

art_df = pd.DataFrame(all_articles)
print(f"Extracted {len(art_df):,} individual articles")
print(art_df[["word_count", "char_len"]].describe())

short_articles = art_df[art_df["word_count"] < 50]
print(f"Articles < 50 words: {len(short_articles)} ({len(short_articles)/len(art_df)*100:.1f}%)")

# Formatting checks
html_count = art_df["text"].str.contains(r"<[^>]+>", regex=True).sum()
url_count = art_df["text"].str.contains(r"https?://", regex=True).sum()
encoding_issues = art_df["text"].str.contains(r"â€|Ã|Â|â€™|â€œ|â€", regex=True).sum()
print(f"Formatting issues - HTML: {html_count}, URLs: {url_count}, Encoding: {encoding_issues}")

art_df["sent_count"] = art_df["text"].apply(lambda x: len(re.split(r'[.!?]+', x)))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Multi-News Dataset — EDA Visualizations", fontsize=16, fontweight="bold")

axes[0, 0].hist(df["article_count"], bins=range(1, df["article_count"].max() + 2),
                edgecolor="black", alpha=0.7, color="#4C72B0")
axes[0, 0].set_xlabel("Number of Articles per Cluster")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Articles per Cluster Distribution")
axes[0, 0].axvline(df["article_count"].median(), color="red", linestyle="--",
                    label=f"Median: {df['article_count'].median():.0f}")
axes[0, 0].legend()

axes[0, 1].hist(df["doc_word_count"], bins=50, edgecolor="black", alpha=0.7, color="#55A868")
axes[0, 1].set_xlabel("Word Count (Full Cluster)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Cluster Total Word Count")
axes[0, 1].axvline(df["doc_word_count"].median(), color="red", linestyle="--",
                    label=f"Median: {df['doc_word_count'].median():,.0f}")
axes[0, 1].legend()

axes[0, 2].hist(art_df["word_count"].clip(upper=2000), bins=50,
                edgecolor="black", alpha=0.7, color="#C44E52")
axes[0, 2].set_xlabel("Word Count (Individual Article)")
axes[0, 2].set_ylabel("Frequency")
axes[0, 2].set_title("Individual Article Word Count")
axes[0, 2].axvline(art_df["word_count"].median(), color="blue", linestyle="--",
                    label=f"Median: {art_df['word_count'].median():,.0f}")
axes[0, 2].legend()

axes[1, 0].hist(df["summary_word_count"], bins=50, edgecolor="black", alpha=0.7, color="#8172B2")
axes[1, 0].set_xlabel("Word Count")
axes[1, 0].set_ylabel("Frequency")
axes[1, 0].set_title("Summary Word Count")
axes[1, 0].axvline(df["summary_word_count"].median(), color="red", linestyle="--",
                    label=f"Median: {df['summary_word_count'].median():,.0f}")
axes[1, 0].legend()

axes[1, 1].hist(art_df["char_len"], bins=50, edgecolor="black", alpha=0.7, color="#CCB974")
axes[1, 1].set_xlabel("Character Length")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Individual Article Character Length")
axes[1, 1].set_yscale("log")

axes[1, 2].hist(art_df["sent_count"].clip(upper=100), bins=50,
                edgecolor="black", alpha=0.7, color="#64B5CD")
axes[1, 2].set_xlabel("Sentence Count")
axes[1, 2].set_ylabel("Frequency")
axes[1, 2].set_title("Sentences per Article")
axes[1, 2].axvline(art_df["sent_count"].median(), color="red", linestyle="--",
                    label=f"Median: {art_df['sent_count'].median():.0f}")
axes[1, 2].legend()

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "multinews_eda_visualizations.png")
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualizations to: {viz_path}")

# Preprocessing: split clusters, normalize whitespace, clean summary formatting
preprocessed_articles = []
for idx, row in df.iterrows():
    articles = row["document"].split("|||||")
    summary = row["summary"].strip()
    
    if summary.startswith("–") or summary.startswith("—"):
        summary = summary[1:].strip()
    elif summary.startswith("- "):
        summary = summary[2:].strip()
    
    for art_idx, article_text in enumerate(articles):
        article_text = article_text.strip()
        article_text = re.sub(r'\n{3,}', '\n\n', article_text)
        article_text = re.sub(r'[ \t]+', ' ', article_text)
        article_text = re.sub(r' \n', '\n', article_text)
        
        if not article_text or len(article_text.split()) < 10:
            continue
        
        doc_id = f"mn_{idx:05d}_{art_idx:02d}"
        content_hash = hashlib.md5(article_text.encode()).hexdigest()[:12]
        
        preprocessed_articles.append({
            "doc_id": doc_id,
            "cluster_id": f"mn_cluster_{idx:05d}",
            "article_index": art_idx,
            "total_articles_in_cluster": len([a for a in articles if a.strip() and len(a.strip().split()) >= 10]),
            "text": article_text,
            "word_count": len(article_text.split()),
            "char_count": len(article_text),
            "summary": summary,
            "content_hash": content_hash,
        })

clean_df = pd.DataFrame(preprocessed_articles)
print(f"Cleaned articles: {len(clean_df):,} (dropped {len(art_df) - len(clean_df):,} short/empty)")

hash_dupes = clean_df["content_hash"].duplicated()
dupe_count = hash_dupes.sum()
print(f"Exact duplicate articles by hash: {dupe_count}")

cluster_sizes = clean_df.groupby("cluster_id").size()
empty_after = (clean_df["text"].str.strip() == "").sum()
assert empty_after == 0, "Found empty articles after cleaning"
print(f"Clusters: {clean_df['cluster_id'].nunique():,}, mean articles/cluster: {cluster_sizes.mean():.2f}")

# Sample clusters for manual inspection
multi_clusters = clean_df[clean_df["total_articles_in_cluster"] >= 3]
sample_cluster_ids = multi_clusters["cluster_id"].unique()[:3]

for cluster_id in sample_cluster_ids:
    cluster = clean_df[clean_df["cluster_id"] == cluster_id]
    print(f"\nCluster: {cluster_id} ({len(cluster)} articles)")
    print(f"Summary: {cluster.iloc[0]['summary'][:150]}...")
    for _, art in cluster.head(2).iterrows():
        sentences = re.split(r'(?<=[.!?])\s+', art['text'])
        print(f"  [{art['doc_id']}] {sentences[0][:100]}...")

# Export cleaned data and statistics
csv_path = os.path.join(OUTPUT_DIR, "multinews_cleaned_sample.csv")
clean_df.to_csv(csv_path, index=False)

jsonl_path = os.path.join(OUTPUT_DIR, "multinews_cleaned_sample.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in clean_df.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

stats = {
    "dataset": "Multi-News",
    "sample_size": len(df),
    "total_train_size": len(ds["train"]),
    "total_articles_extracted": len(clean_df),
    "clusters_in_sample": clean_df["cluster_id"].nunique(),
    "articles_per_cluster_mean": round(cluster_sizes.mean(), 2),
    "articles_per_cluster_median": float(cluster_sizes.median()),
    "article_word_count_mean": round(clean_df["word_count"].mean(), 0),
    "article_word_count_median": float(clean_df["word_count"].median()),
    "exact_duplicates_found": int(dupe_count),
    "missing_values": int(missing.sum()),
    "has_source_metadata": False,
    "has_date_metadata": False,
    "has_contradiction_labels": False,
}
stats_path = os.path.join(OUTPUT_DIR, "multinews_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Exported cleaned data to {csv_path} and stats to {stats_path}")
