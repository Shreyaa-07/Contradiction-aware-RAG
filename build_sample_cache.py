import hashlib
import json
import os
import re
import sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

base_dir = Path(__file__).resolve().parent
out_dir = base_dir / "output" / "multinews"
out_dir.mkdir(parents=True, exist_ok=True)
csv_file = out_dir / "multinews_cleaned_sample.csv"

if not csv_file.exists():
    print("Generating multinews_cleaned_sample.csv from local cache...")
    src_path = hf_hub_download(repo_id="alexfabbri/multi_news", filename="data/train.src.cleaned", repo_type="dataset")
    tgt_path = hf_hub_download(repo_id="alexfabbri/multi_news", filename="data/train.tgt", repo_type="dataset")
    
    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
        src_lines = [next(f) for _ in range(2000)]
    with open(tgt_path, "r", encoding="utf-8", errors="replace") as f:
        tgt_lines = [next(f) for _ in range(2000)]
        
    rows = []
    for idx, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
        articles = src_line.strip().split("|||||")
        summary = tgt_line.strip()
        if summary.startswith(("-", "–", "—")):
            summary = summary[1:].strip()
        for art_idx, art_text in enumerate(articles):
            art_text = re.sub(r"\s+", " ", art_text).strip()
            if len(art_text.split()) >= 10:
                rows.append({
                    "doc_id": f"mn_{idx:05d}_{art_idx:02d}",
                    "cluster_id": f"mn_cluster_{idx:05d}",
                    "text": art_text,
                    "word_count": len(art_text.split()),
                    "summary": summary,
                    "content_hash": hashlib.md5(art_text.encode()).hexdigest()[:12]
                })
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    print(f"Successfully generated {len(df)} articles to {csv_file}")
else:
    print(f"File {csv_file} already exists.")

