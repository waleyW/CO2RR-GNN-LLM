#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-threaded GPT-5 Alloy Synthesis Extractor
----------------------------------------------
Function:
1. Process multiple .txt literature files in parallel;
2. Call GPT-5 to extract "alloy synthesis methods";
3. Output result.txt for each file + a summary JSON;
4. Includes detailed debugging output (input length, response preview, error logs).

Example usage:
python gpt5_alloy_method_extractor.py \
  --input_directory /path/to/texts \
  --output_directory /path/to/results_gpt5 \
  --threads 5
"""

import os
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ======== Initialize GPT-5 client ========
load_dotenv('/GPT_api/api.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ======== Global prompt ========
SYNTHESIS_PROMPT = """
You are an information extractor. Use ONLY information explicitly stated in the paper, no guessing or hallucination. If something is not mentioned, write "Not provided". Extract synthesis-strategy information relevant for precursor ordering and high-level method selection. Return a JSON with: { "material": "", "precursors": [{"name":"", "role":"metal_A or metal_B", "notes":""}], "precursor_properties": { "solubility":{"A":"", "B":""}, "hydrolysis_rate":{"A":"", "B":""}, "complexation":{"A":"", "B":""}, "thermal_behavior":{"A":"", "B":""}, "redox_behavior":{"A":"", "B":""} }, "synthesis_method_label": "", "addition_sequence": "A→B / B→A / Simultaneous / Not provided", "addition_sequence_reason": "", "structure_or_morphology": "", "active_site_related_comments": "", "author_explanations": ["quote1","quote2","quote3"] }. Paper: <<<INSERT PAPER TEXT>>> 

"""


# ======== GPT inference function ========
def gpt5_inference(prompt, text, model="gpt-5", max_tokens=800):
    """
    Robust GPT-5 inference with fallback and retry
    """
    import re
    MAX_INPUT_CHARS = 48000  # Safe input limit for GPT-5 (approx. 32k tokens)

    # ---- Step 1: Truncate overly long input ----
    if len(text) > MAX_INPUT_CHARS:
        print(f"⚠️ Input too long ({len(text)} chars). Truncating...")
        text = text[:MAX_INPUT_CHARS]

    full_prompt = f"Article:\n{text}\n\nTask:\n{prompt}"
    print(f"📏 Effective input length: {len(full_prompt)} chars")

    messages = [
        {"role": "system", "content": "You are an expert in alloy synthesis extraction."},
        {"role": "user", "content": full_prompt},
    ]

    # ---- Step 2: Attempt multiple requests ----
    for attempt in range(3):
        try:
            # Automatically support new / legacy parameter formats
            kwargs = {}
            try:
                kwargs["max_output_tokens"] = max_tokens
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            except TypeError:
                kwargs = {"max_completion_tokens": max_tokens}
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)

            result = response.choices[0].message.content.strip()

            if not result or result.strip() in ["...", "…"]:
                print(f" Empty or truncated response on attempt {attempt+1}, retrying...")
                time.sleep(2)
                continue

            # ---- Step 3: Print response information ----
            print(f" GPT result length: {len(result)} chars")
            print(f" Preview: {result[:120]}...")
            return result

        except Exception as e:
            print(f" GPT-5 attempt {attempt+1} failed: {e}")
            time.sleep(3)

    print(" All retries failed or response empty.")
    return None


# ======== Single-file processing ========
def process_single_file(file_path, output_dir):
    try:
        print(f"\n Processing {file_path}")
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 50:
            print(f" {file_path.name} is empty or too short, skipping.")
            return {"id": Path(file_path).name, "method_output": "Empty input"}

        result = gpt5_inference(SYNTHESIS_PROMPT, text)
        if not result:
            print(f" No valid result for {file_path}")
            return {"id": Path(file_path).name, "method_output": "Empty result"}

        output_file = Path(output_dir) / Path(file_path).name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())

        print(f" Saved {output_file.name}")
        return {"id": Path(file_path).name, "method_output": result}

    except Exception as e:
        print(f" Error processing {file_path}: {e}")
        return {"id": Path(file_path).name, "method_output": f"Error: {e}"}


# ======== Main function (multi-threaded) ========
def process_files(input_directory, output_directory, max_threads=5):
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([f for f in input_dir.glob("*.txt")])
    total_files = len(txt_files)
    print(f" Found {total_files} text files in {input_dir}")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_file = {executor.submit(process_single_file, f, output_dir): f for f in txt_files}

        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f" [{i}/{total_files}] {file_path.name} done.")
            except Exception as e:
                print(f" [{i}/{total_files}] {file_path.name} failed: {e}")

    # Save summary JSON
    summary_file = output_dir / "all_results.json"
    with open(summary_file, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    print(f"\n Finished processing {total_files} files in {duration:.2f}s.")
    print(f" Results saved to {summary_file}")


# ======== CLI ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded GPT-5 alloy synthesis extractor.")
    parser.add_argument("--input_directory", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save results")
    parser.add_argument("--threads", type=int, default=5, help="Number of parallel threads (default=5)")
    args = parser.parse_args()

    process_files(args.input_directory, args.output_directory, args.threads)
