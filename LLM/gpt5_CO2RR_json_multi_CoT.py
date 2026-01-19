#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================
# Load OpenAI key
# =====================================================
load_dotenv('/nesi/nobackup/uoa04081/wxy/model/GPT_api/api.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEBUG = True  # 想看 GPT 原始输出时，改为 True

# =====================================================
# Robust JSON extraction
# =====================================================

def extract_json(text):
    """
    提取最外层 JSON，使用 { } 配对计数器。
    只返回一个 JSON 字符串（最外层结构）。
    """
    stack = 0
    start = None

    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0:
                start = i
            stack += 1

        elif ch == '}':
            stack -= 1
            if stack == 0 and start is not None:
                return text[start:i+1]

    return None  # 根本没找到 JSON


# =====================================================
# Utility
# =====================================================

def compress_list(lst):
    return "; ".join([x.strip() for x in lst])

def compress_json(data):
    return {
        "EXP": compress_list(data["experimental_observation"]),
        "STORY": data["brain_A_story"],
        "TARGETS": compress_list(data["brain_B_targets"]),
        "TASKS": compress_list(data["brain_C_tasks"]),
    }

# =====================================================
# Prompt template
# =====================================================

LOGIC_PROMPT = """
You MUST output ONLY one JSON object.
NO explanations, NO markdown, NO code fences.

ONLY THIS EXACT STRUCTURE:

{
  "system_id": "<<<SYS>>>",
  "variants": [
    {
      "variant_id": "A",
      "logic_chain": {
        "mechanistic_reasoning": [...],
        "minimal_computation": [...]
      }
    },
    {
      "variant_id": "B",
      "logic_chain": {
        "mechanistic_reasoning": [...],
        "minimal_computation": [...]
      }
    },
    {
      "variant_id": "C",
      "logic_chain": {
        "mechanistic_reasoning": [...],
        "minimal_computation": [...]
      }
    }
  ]
}

TASK:
Use ONLY the information below.
Generate 6–10 sequential causal mechanistic steps per variant.
Generate concrete computational tasks (DFT, NEB, AIMD, PDOS, charge, kinetics).

EXPERIMENTAL OBSERVATIONS:
<<<EXP>>>

MECHANISTIC STORY:
<<<STORY>>>

MECHANISTIC TARGETS:
<<<TARGETS>>>

COMPUTATION TASKS:
<<<TASKS>>>

If you produce any text outside the JSON, delete it yourself.
"""

# =====================================================
# GPT call
# =====================================================

def call_gpt(prompt):
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=3500
    )
    return resp.choices[0].message.content


# =====================================================
# Processing
# =====================================================

def process_one_file(filepath, out_dir):
    name = os.path.basename(filepath).replace(".json", "")
    print(f"[RUN] {name}")

    # ---- load JSON ----
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[ERROR] cannot load {name}: {e}")
        return

    data = compress_json(raw)

    prompt = LOGIC_PROMPT \
        .replace("<<<EXP>>>", data["EXP"]) \
        .replace("<<<STORY>>>", data["STORY"]) \
        .replace("<<<TARGETS>>>", data["TARGETS"]) \
        .replace("<<<TASKS>>>", data["TASKS"]) \
        .replace("<<<SYS>>>", name)

    # 3 attempts
    for attempt in range(1, 4):

        try:
            out = call_gpt(prompt)
        except Exception as e:
            print(f"[GPT ERROR] {name} attempt={attempt}: {e}")
            continue

        if DEBUG:
            print("\n===== GPT RAW =====\n")
            print(out)
            print("\n====================\n")

        # extract JSON
        js_text = extract_json(out)

        if js_text is None:
            print(f"[NO JSON] {name} attempt={attempt}")
            continue

        # validate JSON
        try:
            js = json.loads(js_text)

            # save
            out_path = os.path.join(out_dir, f"{name}.logic_chains.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)

            print(f"[OK] {name}")
            return

        except Exception as e:
            print(f"[PARSE FAIL] {name} attempt={attempt}: {e}")

    print(f"[FAIL] {name}")


# =====================================================
# Folder
# =====================================================

def process_folder(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".json")])

    for f in files:
        process_one_file(os.path.join(in_dir, f), out_dir)


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    process_folder(args.json_dir, args.output_dir)
