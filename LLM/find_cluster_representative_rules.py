#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

JSONL_PATH = "syn_3.jsonl"
CLUSTER_CSV = "rule_clusters.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5   

# ============================
# 1. Load rules
# ============================
rules = {}
texts = {}

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rid = r["global_rule_id"]
        rules[rid] = r
        texts[rid] = (
            r.get("rule_name", "") + ". " +
            r.get("stage_1_precursor_rule", {}).get("principle", "") + ". " +
            r.get("stage_2_transformation_rule", {}).get("principle", "") + ". " +
            r.get("boundary_conditions", "")
        )

# ============================
# 2. Load cluster labels
# ============================
dfc = pd.read_csv(CLUSTER_CSV)
cluster_map = dict(zip(dfc.global_rule_id, dfc.cluster_id))

# group by cluster
clusters = {}
for rid, cid in cluster_map.items():
    if cid == -1:
        continue
    clusters.setdefault(cid, []).append(rid)

print(f"[INFO] Found {len(clusters)} clusters (excluding noise)")

# ============================
# 3. Embedding
# ============================
model = SentenceTransformer(MODEL_NAME)
all_ids = list(texts.keys())
all_texts = [texts[i] for i in all_ids]
emb = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=True)
emb_map = dict(zip(all_ids, emb))

# ============================
# 4. Find medoids
# ============================
cluster_reps = []

for cid, ids in clusters.items():
    vecs = np.array([emb_map[i] for i in ids])
    dist = cosine_distances(vecs)
    mean_dist = dist.mean(axis=1)

    order = np.argsort(mean_dist)[:TOP_K]
    rep_ids = [ids[i] for i in order]

    for rank, rid in enumerate(rep_ids):
        cluster_reps.append({
            "cluster_id": cid,
            "rank_in_cluster": rank + 1,
            "global_rule_id": rid,
            "rule_name": rules[rid].get("rule_name", ""),
            "source_file": rules[rid].get("source_file", "")
        })

df = pd.DataFrame(cluster_reps)
df.to_csv("cluster_representative_rules.csv", index=False)
print("[INFO] Saved cluster_representative_rules.csv")
