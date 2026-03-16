#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer

# ============================
# Parameters
# ============================
JSONL_PATH = "syn_3.jsonl"          
CLUSTER_CSV = "rule_clusters.csv"   
OUT_CSV = "rule_umap_vis.csv"       

MODEL_NAME = "all-MiniLM-L6-v2"

# ============================
# 1. Load rules and build text
# ============================
rules = []
texts = []
ids = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)

        text = (
            r.get("rule_name", "") + ". " +
            r.get("stage_1_precursor_rule", {}).get("principle", "") + ". " +
            r.get("stage_2_transformation_rule", {}).get("principle", "") + ". " +
            r.get("boundary_conditions", "")
        ).strip()

        texts.append(text)
        ids.append(r["global_rule_id"])

print(f"[INFO] Loaded {len(texts)} rules")

# ============================
# 2. Load cluster labels
# ============================
df_cluster = pd.read_csv(CLUSTER_CSV)
cluster_map = dict(zip(df_cluster.global_rule_id, df_cluster.cluster_id))

cluster_ids = [cluster_map[i] for i in ids]

# ============================
# 3. Embedding
# ============================
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True
)

print("[INFO] Embedding finished")

# ============================
# 4. UMAP (2D, visualization only)
# ============================
reducer = umap.UMAP(
    n_neighbors=30,
    n_components=2,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

coords = reducer.fit_transform(embeddings)

print("[INFO] UMAP finished")

# ============================
# 5. Save CSV
# ============================
df_vis = pd.DataFrame({
    "umap_x": coords[:, 0],
    "umap_y": coords[:, 1],
    "cluster_id": cluster_ids,
    "global_rule_id": ids
})

df_vis.to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved {OUT_CSV}")
