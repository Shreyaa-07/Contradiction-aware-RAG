"""
EDA and preprocessing for the VitaminC dataset.
Paper: Schuster, Fisch & Barzilay (NAACL 2021) — arXiv:2104.10850
"""

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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "vitaminc")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compatibility helper for environments with Python 3.14 / newer HF hub
try:
    import huggingface_hub.utils._hf_uris as _hf_uris
    _orig_parse = _hf_uris._parse_repo_body
    def _compat_parse(location, type_, raw):
        try:
            return _orig_parse(location, type_, raw=raw)
        except _hf_uris.HfUriError:
            at_idx = location.find("@")
            if at_idx != -1 and location[:at_idx].count("/") == 0:
                repo_id = location[:at_idx]
                rev_and_path = location[at_idx + 1 :]
                slash_idx = rev_and_path.find("/")
                rev = rev_and_path if slash_idx == -1 else rev_and_path[:slash_idx]
                path = "" if slash_idx == -1 else rev_and_path[slash_idx + 1 :]
                return _hf_uris.HfUri(type=type_, id=repo_id, revision=rev, path_in_repo=path, _raw=raw)
            raise
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

# Load VitaminC dataset
ds = load_dataset("tals/vitaminc")
for split_name in ds:
    print(f"{split_name}: {len(ds[split_name]):,} records")

# Sample 5,000 records for EDA
SAMPLE_SIZE = 5000
sample = ds["train"].shuffle(seed=42).select(range(min(SAMPLE_SIZE, len(ds["train"]))))
df = sample.to_pandas()
print(f"Sample shape: {df.shape}")

# Data quality checks
missing = df.isnull().sum()
empty_strings = (df.select_dtypes(include=["object"]).fillna("").astype(str).apply(lambda col: col.str.strip() == "")).sum()
print("Missing values per column:\n", missing)
print("Empty string records per column:\n", empty_strings)

full_dupes = df.duplicated().sum()
pair_dupes = df.duplicated(subset=["claim", "evidence"]).sum()
print(f"Duplicates - Full rows: {full_dupes}, (claim, evidence) pairs: {pair_dupes}")

label_col = "label"
label_counts = df[label_col].value_counts()
print("Label distribution in sample:\n", label_counts)

if "revision_type" in df.columns:
    print("Revision type distribution:\n", df["revision_type"].value_counts())

if "case_id" in df.columns:
    case_sizes = df.groupby("case_id").size()
    print(f"Unique case_ids: {df['case_id'].nunique():,}, mean records per case: {case_sizes.mean():.2f}")

# Text length analysis
df["claim_word_count"] = df["claim"].astype(str).str.split().str.len()
df["evidence_word_count"] = df["evidence"].astype(str).str.split().str.len()
df["evidence_sent_count"] = df["evidence"].astype(str).apply(
    lambda x: len(re.split(r'[.!?]+', x.strip())) if x.strip() else 0
)

print("Text length statistics:")
print(df[["claim_word_count", "evidence_word_count", "evidence_sent_count"]].describe())

if "page" in df.columns:
    print(f"Unique Wikipedia pages in sample: {df['page'].nunique():,}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("VitaminC Dataset — EDA Visualizations", fontsize=16, fontweight="bold")

if label_col in df.columns:
    colors = {"SUPPORTS": "#55A868", "REFUTES": "#C44E52", "NOT ENOUGH INFO": "#8172B2"}
    bar_colors = [colors.get(str(l), "#4C72B0") for l in label_counts.index]
    axes[0, 0].bar(label_counts.index.astype(str), label_counts.values,
                    color=bar_colors, edgecolor="black", alpha=0.8)
    axes[0, 0].set_xlabel("Label")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Label Distribution")
    axes[0, 0].tick_params(axis='x', rotation=15)

axes[0, 1].hist(df["claim_word_count"], bins=40, edgecolor="black", alpha=0.7, color="#4C72B0")
axes[0, 1].set_xlabel("Word Count")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Claim Word Count Distribution")
axes[0, 1].axvline(df["claim_word_count"].median(), color="red", linestyle="--",
                    label=f"Median: {df['claim_word_count'].median():.0f}")
axes[0, 1].legend()

axes[0, 2].hist(df["evidence_word_count"].clip(upper=150), bins=40,
                edgecolor="black", alpha=0.7, color="#55A868")
axes[0, 2].set_xlabel("Word Count")
axes[0, 2].set_ylabel("Frequency")
axes[0, 2].set_title("Evidence Word Count Distribution")
axes[0, 2].axvline(df["evidence_word_count"].median(), color="red", linestyle="--",
                    label=f"Median: {df['evidence_word_count'].median():.0f}")
axes[0, 2].legend()

if "revision_type" in df.columns:
    rev_counts = df["revision_type"].value_counts()
    axes[1, 0].bar(rev_counts.index.astype(str), rev_counts.values,
                    color="#CCB974", edgecolor="black", alpha=0.8)
    axes[1, 0].set_xlabel("Revision Type")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Revision Type Distribution")
    axes[1, 0].tick_params(axis='x', rotation=45)

if "case_id" in df.columns:
    axes[1, 1].hist(case_sizes, bins=range(1, case_sizes.max() + 2),
                    edgecolor="black", alpha=0.7, color="#C44E52")
    axes[1, 1].set_xlabel("Records per Case ID")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Contrastive Group Sizes")

if label_col in df.columns and "revision_type" in df.columns:
    ct = pd.crosstab(df["revision_type"].astype(str), df[label_col].astype(str))
    ct_norm = ct.div(ct.sum(axis=1), axis=0) * 100
    im = axes[1, 2].imshow(ct_norm.values, aspect="auto", cmap="YlOrRd")
    axes[1, 2].set_xticks(range(len(ct_norm.columns)))
    axes[1, 2].set_xticklabels(ct_norm.columns, rotation=30)
    axes[1, 2].set_yticks(range(len(ct_norm.index)))
    axes[1, 2].set_yticklabels(ct_norm.index)
    axes[1, 2].set_title("Label % by Revision Type")
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

plt.tight_layout()
viz_path = os.path.join(OUTPUT_DIR, "vitaminc_eda_visualizations.png")
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved visualizations to: {viz_path}")

# Preprocessing: normalize whitespace, encode labels, create clean dataset
preprocessed = []
issues_found = 0

for idx, row in df.iterrows():
    claim = str(row.get("claim", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    label = str(row.get("label", "")).strip()
    
    claim = re.sub(r'\s+', ' ', claim)
    evidence = re.sub(r'\s+', ' ', evidence)
    
    if not claim or not evidence or claim == "nan" or evidence == "nan":
        issues_found += 1
        continue
    
    label_map = {
        "SUPPORTS": "SUPPORTS",
        "REFUTES": "REFUTES",
        "NOT ENOUGH INFO": "NOT_ENOUGH_INFO",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "0": "SUPPORTS",
        "1": "REFUTES",
        "2": "NOT_ENOUGH_INFO",
    }
    label_clean = label_map.get(str(label).upper().strip(), str(label))
    record_id = str(row.get("unique_id", f"vc_{idx:06d}"))
    
    preprocessed.append({
        "record_id": record_id,
        "case_id": str(row.get("case_id", "")),
        "claim": claim,
        "evidence": evidence,
        "label": label_clean,
        "page": str(row.get("page", "")),
        "wiki_revision_id": str(row.get("wiki_revision_id", "")),
        "revision_type": str(row.get("revision_type", "")),
        "FEVER_id": str(row.get("FEVER_id", "")),
        "claim_word_count": len(claim.split()),
        "evidence_word_count": len(evidence.split()),
    })

clean_df = pd.DataFrame(preprocessed)
print(f"Cleaned records: {len(clean_df):,} (dropped {issues_found} empty/invalid)")
print("Cleaned label distribution:\n", clean_df["label"].value_counts())

dupes = clean_df.duplicated(subset=["claim", "evidence"]).sum()
print(f"Duplicate (claim, evidence) pairs after cleaning: {dupes}")

# Contrastive pair inspection
if "case_id" in clean_df.columns:
    contrastive = clean_df.groupby("case_id").apply(lambda g: g["label"].nunique() > 1)
    num_contrastive = contrastive.sum()
    print(f"Contrastive case groups (multiple labels): {num_contrastive} ({num_contrastive/len(contrastive)*100:.1f}%)")
    
    contrastive_ids = contrastive[contrastive].index[:2]
    for cid in contrastive_ids:
        group = clean_df[clean_df["case_id"] == cid]
        print(f"\nContrastive group {cid}:")
        for _, row in group.iterrows():
            print(f"  [{row['label']}] Claim: {row['claim'][:80]}")
            print(f"    Evidence: {row['evidence'][:80]}")

# Export cleaned sample and statistics
csv_path = os.path.join(OUTPUT_DIR, "vitaminc_cleaned_sample.csv")
clean_df.to_csv(csv_path, index=False)

jsonl_path = os.path.join(OUTPUT_DIR, "vitaminc_cleaned_sample.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in clean_df.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

stats = {
    "dataset": "VitaminC",
    "sample_size": int(len(df)),
    "total_train_size": int(len(ds["train"])),
    "cleaned_records": int(len(clean_df)),
    "label_distribution": clean_df["label"].value_counts().to_dict(),
    "unique_case_ids": int(clean_df["case_id"].nunique()),
    "unique_pages": int(clean_df["page"].nunique()),
    "claim_word_count_mean": round(clean_df["claim_word_count"].mean(), 1),
    "evidence_word_count_mean": round(clean_df["evidence_word_count"].mean(), 1),
    "has_contradiction_labels": True,
    "has_source_metadata": True,
    "has_date_metadata": True,
    "license": "CC BY-SA 3.0",
}
stats_path = os.path.join(OUTPUT_DIR, "vitaminc_stats.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Exported cleaned data to {csv_path} and stats to {stats_path}")
