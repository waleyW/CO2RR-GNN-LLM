#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历父文件夹下所有子文件夹，读取所有 txt 文件中的 JSON 数据，
仅保留 is_article 和 is_alloy 同时为 True 的记录，
并添加自增 entry 编号与源文件名，输出为统一 CSV。
"""

import json
import argparse
import pandas as pd
from pathlib import Path

# ========== 参数解析 ==========
parser = argparse.ArgumentParser(
    description="Extract filtered JSON (is_article & is_alloy True) and merge into one CSV."
)
parser.add_argument("parent_dir", help="Parent directory containing subdirectories with txt files")
parser.add_argument("--out", default="all_data_filtered.csv", help="Output CSV file name")
parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
args = parser.parse_args()

# 固定列顺序
FIXED_KEYS = [
    "is_article",
    "is_alloy",
    "metal_1",
    "metal_2",
    "precursor_1",
    "precursor_2",
    "solvent_environment",
    "synthesis_method",
    "acidity_condition",
    "reaction_type",
]


def parse_json_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        if '{' in line:
            try:
                start = line.index('{')
                end = line.rindex('}') + 1
                return json.loads(line[start:end])
            except Exception:
                pass
    return None


def read_json_file(filepath):
    """读取一个文件，返回其中的 JSON 对象列表"""
    data_list = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"⚠️ 无法读取文件 {filepath}: {e}")
        return data_list

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    for line in content.split("\n"):
        obj = parse_json_line(line)
        if obj:
            data_list.append(obj)
    return data_list


def collect_all_data(parent_dir, pattern="*.txt"):
    parent_path = Path(parent_dir)
    if not parent_path.exists():
        raise FileNotFoundError(f"❌ 目录不存在: {parent_dir}")

    all_data = []
    file_count = 0

    for txt_file in parent_path.rglob(pattern):
        if txt_file.is_file():
            rel_path = txt_file.relative_to(parent_path)
            data = read_json_file(txt_file)
            if data:
                for obj in data:
                    obj["_source_file"] = str(rel_path)  # 🔹 添加来源文件名
                all_data.extend(data)
                file_count += 1
                print(f"📖 {rel_path}: 读取 {len(data)} 条记录")

    print(f"\n📊 共读取 {file_count} 个文件，总计 {len(all_data)} 条 JSON 记录。")
    return all_data


print("=" * 80)
print("🔍 开始提取并筛选 JSON 数据 ...")
print("=" * 80)

all_data = collect_all_data(args.parent_dir, args.pattern)
if not all_data:
    raise SystemExit(f"❌ 未能从 {args.parent_dir} 中读取任何有效 JSON。")

# 提取固定字段 + 来源文件名
records = []
for obj in all_data:
    record = {key: obj.get(key, "") for key in FIXED_KEYS}
    record["source_file"] = obj.get("_source_file", "")
    records.append(record)

df = pd.DataFrame(records, columns=["source_file"] + FIXED_KEYS)

# 转换布尔并筛选
df["is_article"] = df["is_article"].astype(str).str.lower().eq("true")
df["is_alloy"] = df["is_alloy"].astype(str).str.lower().eq("true")
df_filtered = df[(df["is_article"]) & (df["is_alloy"])].copy()

# 添加 entry 序号
df_filtered.insert(0, "entry", range(1, len(df_filtered) + 1))

# 输出 CSV
df_filtered.to_csv(args.out, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print(f"✅ 已生成筛选后的表格: {args.out}")
print(f"📄 共 {len(df_filtered)} 条记录, {len(df_filtered.columns)} 个字段。")
print("=" * 100)
print(df_filtered.head(10).to_string(index=False))
print("\n✨ 完成！")
