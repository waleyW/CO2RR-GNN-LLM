#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sankey diagram with separate 'Reaction Others' and 'Synthesis Others'
Output: figure_4c_ele_reaction_syn_with_others/figure_4c_ele_reaction_syn_with_2others.png
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ========== I/O ==========
CSV_FILE = Path("ele_alloy_summary.csv")
OUT_DIR = Path("figure_4c_ele_reaction_syn_with_others")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "figure_4c_ele_reaction_syn_with_2others.png"

# ========== 分类函数 ==========
def simplify_reaction(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if "co2" in x or "cor" in x: return "CO₂RR"
    if "her" in x or "hydrogen" in x: return "HER"
    if "oer" in x and "orr" not in x: return "OER"
    if "orr" in x and "oer" not in x: return "ORR"
    if "nrr" in x or "n2" in x: return "NRR"
    return "Reaction Others"

def simplify_method(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if any(k in x for k in ["hydrothermal", "solvothermal", "microwave"]):
        return "② Hydrothermal/Solvothermal"
    if any(k in x for k in ["electrodeposition", "electrochemical", "anodic", "cathodic"]):
        return "① Electrodeposition/Electrochemical"
    if any(k in x for k in ["co-precip", "chemical reduction", "combustion", "precipitation"]):
        return "④ Chemical Reduction/Co-precipitation"
    if any(k in x for k in ["thermal", "calcination", "pyrolysis", "anneal"]):
        return "③ Thermal/Pyrolysis"
    if any(k in x for k in ["sol-gel", "template", "self"]):
        return "⑤ Sol-gel/Template"
    return "⑥ Synthesis Others"

# ========== 读取并分类 ==========
df = pd.read_csv(CSV_FILE)
df["reaction_simplified"] = df["reaction_type"].apply(simplify_reaction)
df["method_simplified"] = df["synthesis_method"].apply(simplify_method)

# ========== 统计组合关系 ==========
combo = df.groupby(["reaction_simplified", "method_simplified"]).size().reset_index(name="count")

reactions = ["CO₂RR", "HER", "OER", "ORR", "NRR", "Reaction Others"]
methods = ["① Electrodeposition/Electrochemical",
           "② Hydrothermal/Solvothermal",
           "③ Thermal/Pyrolysis",
           "④ Chemical Reduction/Co-precipitation",
           "⑤ Sol-gel/Template",
           "⑥ Synthesis Others"]


nodes = reactions + methods
idx = {name: i for i, name in enumerate(nodes)}

# ========== 颜色 ==========
colors_left = ["#4973A8" if n == "CO₂RR" else "#AFC0D6" for n in reactions]
colors_right = ["#E7C65C" if "Electrodeposition" in n else "#EFD89A" for n in methods]
colors = colors_left + colors_right

# ========== Sankey 数据 ==========
sources, targets, values, link_colors = [], [], [], []

for _, row in combo.iterrows():
    src = row["reaction_simplified"]
    tgt = row["method_simplified"]
    val = row["count"]
    if src in idx and tgt in idx:
        sources.append(idx[src])
        targets.append(idx[tgt])
        values.append(val)
        link_colors.append("#4973A8" if src == "CO₂RR" else "rgba(160,160,160,0.4)")

fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=20, thickness=10,
        line=dict(color="black", width=0.3),
        label=nodes,
        color=colors
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
))

# ========== 布局 ==========
fig.update_layout(
    font=dict(family="/nesi/nobackup/uoa04335/WXY/Software/Font/arial.ttf", size=8, color="black"),
    margin=dict(l=40, r=40, t=10, b=10),
    width=480, height=350,  # Nature half-column
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# ========== 注释 ==========
fig.add_annotation(
    text="Blue/Yellow bar width indicates frequency; thicker link = higher co-occurrence",
    x=0.5, y=-0.08, showarrow=False, xref="paper", yref="paper",
    font=dict(size=7, family="Arial", color="black"),
    align="center"
)

# 垂直标签
fig.add_annotation(
    text="Reaction type", x=-0.10, y=0.5, textangle=270,
    showarrow=False, font=dict(size=8, family="Arial", color="black"), xref="paper", yref="paper"
)
fig.add_annotation(
    text="Synthesis method", x=1.08, y=0.5, textangle=90,
    showarrow=False, font=dict(size=8, family="Arial", color="black"), xref="paper", yref="paper"
)

# ========== 输出 ==========
fig.write_image(str(OUT_FILE), scale=6, width=480, height=350)
print(f"✅ PNG 图已生成: {OUT_FILE}")
