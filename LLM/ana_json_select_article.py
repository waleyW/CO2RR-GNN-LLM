#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traverse all subfolders under a parent directory, read JSON data from all txt files,
retain only records where both is_article and is_alloy are True,
and output them into a unified CSV with an auto-increment entry ID and source filename.
"""

import json
import argparse
import pandas as pd
from pathlib import Path

# ========== Argument parsing ==========
parser = argparse.ArgumentParser(
    description="Extract filtered JSON (is_article & is_alloy True) and merge into one CSV."
)
parser.add_argument("parent_dir", help="Parent directory containing subdirectories with txt files")
parser.add_argument("--out", default="all_data_filtered.csv", help="Output CSV file name")
parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
args = parser.parse_args()

# Fixed column order
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
    """Read a file and return the list of JSON objects contained in it"""
    data_list = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        print(f"⚠️ Unable to read file {filepath}: {e}")
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
        raise FileNotFoundError(f" Directory does not exist: {parent_dir}")

    all_data = []
    file_count = 0

    for txt_file in parent_path.rglob(pattern):
        if txt_file.is_file():
            rel_path = txt_file.relative_to(parent_path)
            data = read_json_file(txt_file)
            if data:
                for obj in data:
                    obj["_source_file"] = str(rel_path)  # 🔹 Add source filename
                all_data.extend(data)
                file_count += 1
                print(f"📖 {rel_path}: read {len(data)} records")

    print(f"\n📊 Total {file_count} files read, {len(all_data)} JSON records in total.")
    return all_data


print("=" * 80)
print("🔍 Start extracting and filtering JSON data ...")
print("=" * 80)

all_data = collect_all_data(args.parent_dir, args.pattern)
if not all_data:
    raise SystemExit(f" No valid JSON data could be read from {args.parent_dir}.")

# Extract fixed fields + source filename
records = []
for obj in all_data:
    record = {key: obj.get(key, "") for key in FIXED_KEYS}
    record["source_file"] = obj.get("_source_file", "")
    records.append(record)

df = pd.DataFrame(records, columns=["source_file"] + FIXED_KEYS)

# Convert boolean values and filter
df["is_article"] = df["is_article"].astype(str).str.lower().eq("true")
df["is_alloy"] = df["is_alloy"].astype(str).str.lower().eq("true")
df_filtered = df[(df["is_article"]) & (df["is_alloy"])].copy()

# Add entry index
df_filtered.insert(0, "entry", range(1, len(df_filtered) + 1))

# Export CSV
df_filtered.to_csv(args.out, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print(f" Filtered table generated: {args.out}")
print(f"📄 {len(df_filtered)} records, {len(df_filtered.columns)} fields.")
