#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sankey diagram: Alloy base type vs synthesis method (with 'Others')
Output: figure_4e_ele_alloy_method_base/figure_4e_ele_alloy_method_base.png
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ---------- I/O ----------
CSV_FILE = Path("ele_alloy_summary.csv")
OUT_DIR = Path("figure_4d_ele_alloy_method_base")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "figure_4d_ele_alloy_method_base.png"

# ---------- 分类函数 ----------
def simplify_method(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if any(k in x for k in ["electrodeposition", "electrochemical", "anodic", "cathodic"]):
        return "① Electrodeposition/Electrochemical"
    if any(k in x for k in ["hydrothermal", "solvothermal", "microwave"]):
        return "② Hydrothermal/Solvothermal"
    if any(k in x for k in ["thermal", "calcination", "pyrolysis", "anneal"]):
        return "③ Thermal/Pyrolysis"
    if any(k in x for k in ["co-precip", "chemical reduction", "combustion", "precipitation"]):
        return "④ Chemical Reduction/Co-precipitation"
    if any(k in x for k in ["sol-gel", "template", "self"]):
        return "⑤ Sol-gel/Template"
    return None  # 不含方法 Others

def simplify_metal(row):
    metals = [str(row["metal_1"]).capitalize(), str(row["metal_2"]).capitalize()]
    if any("Cu" in m for m in metals): return "Cu-base"
    if any("Ni" in m for m in metals): return "Ni-base"
    if any("Pt" in m for m in metals): return "Pt-base"
    if any("Pd" in m for m in metals): return "Pd-base"
    return "Others"

# ---------- 读取与分类 ----------
df = pd.read_csv(CSV_FILE)
df["method_simplified"] = df["synthesis_method"].apply(simplify_method)
df["metal_base"] = df.apply(simplify_metal, axis=1)
df = df.dropna(subset=["method_simplified", "metal_base"])

# ---------- 统计 ----------
combo = df.groupby(["metal_base", "method_simplified"]).size().reset_index(name="count")

metals = ["Cu-base", "Ni-base", "Pt-base", "Pd-base", "Others"]
methods = [
    "① Electrodeposition/Electrochemical",
    "② Hydrothermal/Solvothermal",
    "③ Thermal/Pyrolysis",
    "④ Chemical Reduction/Co-precipitation",
    "⑤ Sol-gel/Template"
]
nodes = metals + methods
idx = {n: i for i, n in enumerate(nodes)}

# ---------- 颜色 ----------
colors_left = ["#AFC0D6", "#97B2C9", "#7F9DBF", "#6685A8", "#C5CFDC"]
colors_right = ["#EFD89A"] * len(methods)
colors = colors_left + colors_right



# ---------- Sankey 数据 ----------
sources, targets, values = [], [], []
for _, r in combo.iterrows():
    if r["metal_base"] in idx and r["method_simplified"] in idx:
        sources.append(idx[r["metal_base"]])
        targets.append(idx[r["method_simplified"]])
        values.append(r["count"])

link_colors = []
for s, t, v in zip(sources, targets, values):
    src_label = nodes[s]
    if src_label == "Cu-base":
        link_colors.append("rgba(79,112,167,0.7)")  # 蓝色更亮一点
    else:
        link_colors.append("rgba(120,120,120,0.4)")
        
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

# ---------- 布局 ----------
fig.update_layout(
    font=dict(family="/nesi/nobackup/uoa04335/WXY/Software/Font/arial.ttf", size=8, color="black"),
    margin=dict(l=20, r=20, t=5, b=25),
    width=315, height=200,   # Nature half-column
    paper_bgcolor="white",
    plot_bgcolor="white"
)

# ---------- 注释 ----------
fig.add_annotation(
    text="*Wider bars → more literature; wider links → stronger co-occurrence",
    x=0.5, y=-0.08, showarrow=False, xref="paper", yref="paper",
    font=dict(size=7, family="Arial", color="black"), align="center"
)
fig.add_annotation(
    text="Alloy combo", x=-0.05, y=0.5, textangle=270,
    showarrow=False, font=dict(size=8, family="Arial", color="black"), xref="paper", yref="paper"
)
fig.add_annotation(
    text="Synthesis method", x=1.05, y=0.5, textangle=90,
    showarrow=False, font=dict(size=8, family="Arial", color="black"), xref="paper", yref="paper"
)

# ---------- 输出 ----------
fig.write_image(str(OUT_FILE), scale=6, width=315, height=200)
print(f"✅ PNG 图已生成: {OUT_FILE}")
