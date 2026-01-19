#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cu-based alloy vs synthesis method — main + SI + all-method version
-------------------------------------------------------------
✅ 连线宽度 ∝ count
✅ 颜色灰→蓝渐变 ∝ count
✅ 越深颜色在线条上层
✅ 图注两行说明：alloy–synthesis method pair
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import matplotlib.colors as mcolors
import numpy as np

# ---------- I/O ----------
CSV_FILE = Path("ele_alloy_summary.csv")
OUT_DIR = Path("figure_4d_ele_cu_alloy_method")
OUT_DIR.mkdir(exist_ok=True)

OUT_TOP10_PNG = OUT_DIR / "figure_4d_ele_cu_alloy_method_top10.png"
OUT_ALL_PNG   = OUT_DIR / "figure_4d_ele_cu_alloy_method_all.png"
OUT_ALL_OTHERS_PNG = OUT_DIR / "figure_4d_ele_cu_alloy_method_all_with_others.png"
OUT_TOP10_CSV = OUT_DIR / "figure_4d_ele_cu_alloy_method_top10.csv"
OUT_ALL_CSV   = OUT_DIR / "figure_4d_ele_cu_alloy_method_all.csv"
OUT_ALL_OTHERS_CSV = OUT_DIR / "figure_4d_ele_cu_alloy_method_all_with_others.csv"

# ---------- 分类 ----------
def simplify_method(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if any(k in x for k in ["electrodeposition","electrochemical","anodic","cathodic"]):
        return "① Electrodeposition/Electrochemical"
    if any(k in x for k in ["hydrothermal","solvothermal","microwave"]):
        return "② Hydrothermal/Solvothermal"
    if any(k in x for k in ["thermal","calcination","pyrolysis","anneal"]):
        return "③ Thermal/Pyrolysis"
    if any(k in x for k in ["co-precip","chemical reduction","combustion","precipitation"]):
        return "④ Chemical Reduction/Co-precipitation"
    if any(k in x for k in ["sol-gel","template","self"]):
        return "⑤ Sol-gel/Template"
    return "⑥ Others"

# ---------- 读取 ----------
df = pd.read_csv(CSV_FILE)
df["method_simplified"] = df["synthesis_method"].apply(simplify_method)

# 保证 Cu 在前
def format_alloy(m1, m2):
    m1, m2 = str(m1), str(m2)
    if "Cu" in m1 and "Cu" not in m2: return f"Cu-{m2}"
    if "Cu" in m2 and "Cu" not in m1: return f"Cu-{m1}"
    return "-".join(sorted([m1, m2]))

df_cu = df[df[["metal_1","metal_2"]].apply(lambda x: any("Cu" in str(i) for i in x), axis=1)].copy()
df_cu["alloy_combo"] = df_cu.apply(lambda r: format_alloy(r["metal_1"], r["metal_2"]), axis=1)
df_cu = df_cu.dropna(subset=["method_simplified"])

# ---------- 统计 ----------
combo_all = df_cu.groupby(["alloy_combo","method_simplified"]).size().reset_index(name="count")
combo_all.to_csv(OUT_ALL_CSV,index=False)

# Top10
top_alloys = combo_all.groupby("alloy_combo")["count"].sum().nlargest(10).index.tolist()
combo_top10 = combo_all[combo_all["alloy_combo"].isin(top_alloys)]
combo_top10.to_csv(OUT_TOP10_CSV,index=False)

# All + Others
combo_all_with_others = combo_all.copy()
combo_all_with_others.to_csv(OUT_ALL_OTHERS_CSV,index=False)

# ---------- 绘图函数 ----------
def plot_sankey(combo_df, alloys_subset, include_others, out_file, fig_height):
    alloys = sorted(alloys_subset)
    methods = [
        "① Electrodeposition/Electrochemical",
        "② Hydrothermal/Solvothermal",
        "③ Thermal/Pyrolysis",
        "④ Chemical Reduction/Co-precipitation",
        "⑤ Sol-gel/Template"
    ]
    if include_others:
        methods.append("⑥ Others")
    nodes = alloys + methods
    idx = {n:i for i,n in enumerate(nodes)}

    # 节点颜色
    colors_left  = ["#AFC0D6"]*len(alloys)
    colors_right = ["#E5C27E"]*len(methods)
    colors = colors_left + colors_right

    # Sankey 数据（按 count 升序排序，浅色在下层，深色在上层）
    combo_sorted = combo_df.sort_values("count")
    src,tgt,val=[],[],[]
    for _,r in combo_sorted.iterrows():
        if r["alloy_combo"] in idx and r["method_simplified"] in idx:
            src.append(idx[r["alloy_combo"]])
            tgt.append(idx[r["method_simplified"]])
            val.append(r["count"])

    # ---------- 颜色深度映射（灰→蓝） ----------
    min_c, max_c = min(val), max(val)
    norm = mcolors.Normalize(vmin=min_c, vmax=max_c)
    # 浅灰更亮，蓝色层次保持
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gray_blue", ["#F0F0F0", "#AFC0D6", "#4F70A7"]
    )
    link_colors = [mcolors.to_rgba(cmap(norm(v))) for v in val]
    # 调整透明度让浅线更轻盈
    link_colors = [
        f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {0.8 * a:.2f})"
        for r,g,b,a in link_colors
    ]

    # ---------- 绘图 ----------
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18,thickness=10,
            line=dict(color="black",width=0.3),
            label=nodes,color=colors
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
            color=link_colors
        )
    ))

    # ---------- 布局 ----------
    fig.update_layout(
        font=dict(family="/nesi/nobackup/uoa04335/WXY/Software/Font/arial.ttf",
                  size=8,color="black"),
        margin=dict(l=30,r=30,t=5,b=45),
        width=315,height=fig_height,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    # 左右标签
    fig.add_annotation(text="Cu alloys",x=-0.06,y=0.5,textangle=270,
                       showarrow=False,font=dict(size=8,family="Arial",color="black"),
                       xref="paper",yref="paper")
    fig.add_annotation(text="Synthesis method",x=1.06,y=0.5,textangle=90,
                       showarrow=False,font=dict(size=8,family="Arial",color="black"),
                       xref="paper",yref="paper")
    # 两行说明文字
    fig.add_annotation(
        text="Link width and color intensity are proportional to <br>the frequency of each alloy–synthesis method pair.",
        x=0.5, y=-0.18, showarrow=False, xref="paper", yref="paper",
        font=dict(size=7, family="Arial", color="black"), align="center"
    )

    fig.write_image(str(out_file),scale=6,width=315,height=fig_height)
    print(f"✅ Sankey 图已生成: {out_file}")

# ---------- 绘图 ----------
plot_sankey(combo_top10, top_alloys, include_others=False,
            out_file=OUT_TOP10_PNG, fig_height=200)
plot_sankey(combo_all, combo_all["alloy_combo"].unique(), include_others=False,
            out_file=OUT_ALL_PNG, fig_height=400)
plot_sankey(combo_all_with_others, combo_all_with_others["alloy_combo"].unique(),
            include_others=True, out_file=OUT_ALL_OTHERS_PNG, fig_height=420)
