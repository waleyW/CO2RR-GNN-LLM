#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 ele_alloy_summary.csv 生成分类表（含 Reaction 与 Synthesis）
输出:
figure_4c_ele_reaction_syn_noothers/SI_reaction_method_categories.csv

列:
Type | Category | Subclasses (原始出现的小类)
"""

import pandas as pd
from pathlib import Path

# ======== 文件路径 ========
CSV_FILE = Path("ele_alloy_summary.csv")
OUT_DIR = Path("figure_4c_ele_reaction_syn_noothers")
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "SI_reaction_method_categories.csv"

# ======== 分类函数 ========
def simplify_reaction(x):
    if not isinstance(x, str): return None
    x = x.lower()
    if "co2" in x or "cor" in x: return "CO₂RR"
    if "her" in x or "hydrogen" in x: return "HER"
    if "oer" in x and "orr" not in x: return "OER"
    if "orr" in x and "oer" not in x: return "ORR"
    if "nrr" in x or "n2" in x: return "NRR"
    return "Others"

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
    return "Others"

# ======== 读取数据 ========
print(f"📂 Reading CSV: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)

if not {"reaction_type", "synthesis_method"}.issubset(df.columns):
    raise ValueError("❌ CSV 文件中必须包含列: reaction_type, synthesis_method")

# ======== 分类 ========
df["reaction_simplified"] = df["reaction_type"].apply(simplify_reaction)
df["method_simplified"] = df["synthesis_method"].apply(simplify_method)

# ======== 统计每类的小类集合 ========
records = []

# Reaction 部分
for cat, sub_df in df.groupby("reaction_simplified"):
    unique_terms = sorted(set(sub_df["reaction_type"].dropna().unique()))
    records.append({
        "Type": "Reaction",
        "Category": cat,
        "Subclasses": ", ".join(unique_terms)
    })

# Method 部分
for cat, sub_df in df.groupby("method_simplified"):
    unique_terms = sorted(set(sub_df["synthesis_method"].dropna().unique()))
    records.append({
        "Type": "Synthesis",
        "Category": cat,
        "Subclasses": ", ".join(unique_terms)
    })

# ======== 输出 ========
out_df = pd.DataFrame(records, columns=["Type", "Category", "Subclasses"])
out_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ SI 分类表生成完成: {OUT_FILE}")
print(out_df.to_string(index=False))
