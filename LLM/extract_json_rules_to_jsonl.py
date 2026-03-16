import os
import json
import re
from glob import glob

TXT_DIR = "syn_3"
OUT_JSONL = "syn_3.jsonl"


# ============================================================
# 1️⃣ Original method: strict bracket-balanced JSON extraction
# (This is the method used to extract the previous 942 rules)
# ============================================================
def extract_json_strict(text):
    blocks = []
    stack = []
    start_idx = None

    for i, ch in enumerate(text):
        if ch in ['{', '[']:
            if not stack:
                start_idx = i
            stack.append(ch)

        elif ch in ['}', ']']:
            if not stack:
                continue
            stack.pop()
            if not stack and start_idx is not None:
                block = text[start_idx:i+1]
                try:
                    parsed = json.loads(block)
                    if isinstance(parsed, list):
                        blocks.extend([x for x in parsed if isinstance(x, dict)])
                    elif isinstance(parsed, dict):
                        blocks.append(parsed)
                except json.JSONDecodeError:
                    pass
                start_idx = None

    return blocks


# ============================================================
# 2️⃣ Fallback method: salvage extraction
# Only used when strict parsing fails
# ============================================================
def clean_text(text):
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    return text


def try_parse_json(candidate):
    open_curly = candidate.count("{")
    close_curly = candidate.count("}")
    if close_curly < open_curly:
        candidate += "}" * (open_curly - close_curly)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def extract_json_salvage(text):
    text = clean_text(text)
    rules = []

    parts = re.split(r"(?=\{)", text)
    for part in parts:
        part = part.strip()
        if not part.startswith("{"):
            continue
        parsed = try_parse_json(part)
        if isinstance(parsed, dict):
            rules.append(parsed)
    return rules


# ============================================================
# 3️⃣ Main workflow
# ============================================================
all_rules = []
global_counter = 1

for path in glob(os.path.join(TXT_DIR, "*.txt")):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # First attempt strict parsing
    rules = extract_json_strict(text)

    # If strict parsing fails, use salvage mode
    if len(rules) == 0:
        rules = extract_json_salvage(text)

    print(f"[INFO] {os.path.basename(path)} -> {len(rules)} rules")

    for r in rules:
        original_rule_id = r.get("rule_id", None)

        r["global_rule_id"] = global_counter
        r["original_rule_id"] = original_rule_id
        r["rule_id"] = (
            str(original_rule_id)
            if original_rule_id is not None
            else f"auto_{global_counter}"
        )
        r["source_file"] = os.path.basename(path)

        all_rules.append(r)
        global_counter += 1


print(f"\n[INFO] Total extracted rules: {len(all_rules)}")

with open(OUT_JSONL, "w", encoding="utf-8") as f:
    for r in all_rules:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[INFO] Saved to {OUT_JSONL}")