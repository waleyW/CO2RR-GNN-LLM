import json
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import pandas as pd


JSONL_PATH = "syn_3.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"  
UMAP_DIM = 10
MIN_CLUSTER_SIZE = 10

rules = []
texts = []
ids = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        rules.append(r)

        text = (
            r.get("rule_name", "") + ". " +
            r.get("stage_1_precursor_rule", {}).get("principle", "") + ". " +
            r.get("stage_2_transformation_rule", {}).get("principle", "") + ". " +
            r.get("boundary_conditions", "")
        )

        texts.append(text)
        ids.append(r["global_rule_id"])

print(f"[INFO] Loaded {len(texts)} rules")


model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, show_progress_bar=True)


reducer = umap.UMAP(
    n_neighbors=50,
    n_components=UMAP_DIM,
    min_dist=0.0,
    random_state=42
)
embeddings_umap = reducer.fit_transform(embeddings)


clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    metric="euclidean"
)

labels = clusterer.fit_predict(embeddings_umap)

df = pd.DataFrame({
    "global_rule_id": ids,
    "cluster_id": labels
})

df.to_csv("rule_clusters_noise50.csv", index=False)
print("[INFO] Saved rule_clusters.csv")


n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_ratio = np.mean(labels == -1)

print(f"[INFO] Clusters: {n_clusters}")
print(f"[INFO] Noise ratio: {noise_ratio:.2f}")
