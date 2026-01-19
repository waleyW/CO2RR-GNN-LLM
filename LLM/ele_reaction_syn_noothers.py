#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nature half-column Sankey diagram (final, no legend, fixed order)
- Reaction type ↔ Synthesis method
- Both labels vertical (same rotation)
- No legend, only explanatory note
- Output folder: figure_4c_ele_reaction_syn_noothers
- Manually fixed node order using explicit y positions
"""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from matplotlib import font_manager, rcParams

# ======== Font ========
font_path = "/nesi/nobackup/uoa04335/WXY/Software/Font/arial.ttf"
font_manager.fontManager.addfont(font_path)
rcParams["font.family"] = "Arial"
rcParams["font.size"] = 8

# ======== Paths ========
INPUT_FILE = "ele_alloy_summary.csv"
OUT_DIR = Path("figure_4c_ele_reaction_syn_noothers")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "figure_4c_ele_reaction_syn_final_CO2RR_noothers_fixed_order"

# ======== Read ========
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["reaction_type", "synthesis_method"])

# ======== Simplify ========
def simplify_reaction(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if "co2" in x or "cor" in x: return "CO₂RR"
    if "her" in x or "hydrogen" in x: return "HER"
    if "oer" in x and "orr" not in x: return "OER"
    if "orr" in x and "oer" not in x: return "ORR"
    if "nrr" in x or "n2" in x: return "NRR"
    return None

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
    return None


df["reaction_simplified"] = df["reaction_type"].apply(simplify_reaction)
df["method_simplified"]   = df["synthesis_method"].apply(simplify_method)
df = df.dropna(subset=["reaction_simplified", "method_simplified"])

# ======== Count & Export ========
combo = df.groupby(["reaction_simplified","method_simplified"]).size().reset_index(name="count")
combo.pivot(index="reaction_simplified", columns="method_simplified", values="count")\
     .fillna(0).astype(int).to_csv(OUT_DIR/"reaction_method_counts_CO2RR_noothers_fixed_order.csv", encoding="utf-8-sig")


# ======== Sankey Data ========
reactions = ["CO₂RR","HER","OER","ORR","NRR"]
methods = [
    "② Hydrothermal/Solvothermal",
    "① Electrodeposition/Electrochemical",
    "④ Chemical Reduction/Co-precipitation",
    "③ Thermal/Pyrolysis",
    "⑤ Sol-gel/Template"
]

# ======== Color ========
colors_left = ["#AFC0D6", "#97B2C9", "#7F9DBF", "#6685A8", "#C5CFDC"]
colors_right = ["#EFD89A"] * len(methods)
colors = colors_left + colors_right
nodes = reactions + methods
idx = {n:i for i,n in enumerate(nodes)}

sources = [idx[r] for r in combo["reaction_simplified"]]
targets = [idx[m] for m in combo["method_simplified"]]
values  = combo["count"].tolist()

# ======== Link colors ========
link_colors = [
    "rgba(73,115,168,0.9)" if r == "CO₂RR" else "rgba(150,150,150,0.1)"
    for r in combo["reaction_simplified"]
]

# ======== Fixed vertical order (y positions) ========
# Left: reactions (top→bottom) | Right: methods (top→bottom)
y_positions = [
    0.95,  # CO₂RR
    0.75,  # HER
    0.55,  # OER
    0.35,  # ORR
    0.15,  # NRR
    0.90,  # ① Hydrothermal/Solvothermal
    0.72,  # ② Electrodeposition/Electrochemical
    0.54,  # ③ Chemical Reduction/Co-precipitation
    0.36,  # ④ Thermal/Pyrolysis
    0.18   # ⑤ Sol-gel/Template
]

# ======== Draw Sankey ========
fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=10, thickness=11,
        label=nodes,
        color=colors,
        line=dict(color="black", width=0.3),
        y=y_positions
    ),
    link=dict(source=sources, target=targets, value=values, color=link_colors)
)])

fig.update_layout(
    font=dict(size=8, family="Arial", color="black"),
    width=315, height=200,  # Nature half-column
    margin=dict(l=20, r=20, t=5, b=25)
)

# ======== Vertical labels (same direction) ========
fig.add_annotation(
    text="Reaction type",
    x=-0.05, y=0.5, xref="paper", yref="paper",
    showarrow=False, font=dict(size=8, family="Arial", color="black"),
    textangle=270, align="center"
)
fig.add_annotation(
    text="Synthesis method",
    x=1.05, y=0.5, xref="paper", yref="paper",
    showarrow=False, font=dict(size=8, family="Arial", color="black"),
    textangle=90, align="center"
)

# ======== Explanation text ========
fig.add_annotation(
    text="*Wider bars → more literature; wider links → stronger co-occurrence",
    x=0.5, y=-0.08, xref="paper", yref="paper",
    showarrow=False, font=dict(size=8, family="Arial",color="black"), align="center"
)

# ======== Export ========
fig.write_image(str(OUT_FILE)+".svg")
fig.write_image(str(OUT_FILE)+".png", scale=6)
print(f"✅ 输出: {OUT_FILE}.(svg/png) & reaction_method_counts_CO2RR_noothers_fixed_order.csv")
