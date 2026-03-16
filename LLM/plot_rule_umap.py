#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ============================
# Font (Nature)
# ============================
font_path = "/Font/arial.ttf"
arial = fm.FontProperties(fname=font_path)

plt.rcParams["font.family"] = arial.get_name()
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================
# Load data
# ============================
df = pd.read_csv("rule_umap_vis.csv")

df_noise = df[df.cluster_id == -1]
df_clustered = df[df.cluster_id != -1]
clusters = sorted(df_clustered.cluster_id.unique())

# ============================
# Color palette (yellow–blue, continuous)
# ============================
cmap = plt.cm.cividis  
norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))

color_map = {
    cid: cmap(norm(cid))
    for cid in clusters
}

# ============================
# Plot
# ============================
fig, ax = plt.subplots(figsize=(3.5, 3.2)) 

# Noise (background)
ax.scatter(
    df_noise.umap_x,
    df_noise.umap_y,
    c="lightgrey",
    s=8,
    alpha=0.35,
    linewidths=0,
    label="Unclustered"
)

# Clusters
for cid in clusters:
    d = df_clustered[df_clustered.cluster_id == cid]
    ax.scatter(
        d.umap_x,
        d.umap_y,
        s=10,
        color=color_map[cid],
        alpha=0.85,
        linewidths=0,
        label=f"Cluster {cid}"
    )

# ============================
# Axes style (box ON)
# ============================
ax.set_xlabel("UMAP-1", fontsize=11, fontproperties=arial)
ax.set_ylabel("UMAP-2", fontsize=11, fontproperties=arial)

ax.tick_params(
    left=False, bottom=False,
    labelleft=False, labelbottom=False
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

# ============================
# Legend (inside box)
# ============================
legend = ax.legend(
    loc="upper right",
    fontsize=8,
    frameon=False,
    markerscale=1.0,
    handletextpad=0.4,
    labelspacing=0.3,
    borderaxespad=0.6
)

# ============================
# Save
# ============================
plt.tight_layout()
plt.savefig(
    "Figure_rule_umap_nature_box_legend.png",
    dpi=600,
    transparent=True
)
plt.show()

print("[INFO] Figure saved as Figure_rule_umap_nature_box_legend.png")
