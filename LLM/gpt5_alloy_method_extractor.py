#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-threaded GPT-5 Alloy Synthesis Extractor
----------------------------------------------
功能：
1. 并行处理多个 .txt 文献文件；
2. 调用 GPT-5 提取“合金合成方法”；
3. 输出每篇文件的 result.txt + 汇总 JSON；
4. 含详细调试输出（输入长度、返回预览、错误日志）。

使用示例：
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

# ======== 初始化 GPT-5 客户端 ========
load_dotenv('/nesi/nobackup/uoa04081/wxy/model/GPT_api/api.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ======== 全局 prompt ========
SYNTHESIS_PROMPT = """
You are an information extractor. Use ONLY information explicitly stated in the paper, no guessing or hallucination. If something is not mentioned, write "Not provided". Extract synthesis-strategy information relevant for precursor ordering and high-level method selection. Return a JSON with: { "material": "", "precursors": [{"name":"", "role":"metal_A or metal_B", "notes":""}], "precursor_properties": { "solubility":{"A":"", "B":""}, "hydrolysis_rate":{"A":"", "B":""}, "complexation":{"A":"", "B":""}, "thermal_behavior":{"A":"", "B":""}, "redox_behavior":{"A":"", "B":""} }, "synthesis_method_label": "", "addition_sequence": "A→B / B→A / Simultaneous / Not provided", "addition_sequence_reason": "", "structure_or_morphology": "", "active_site_related_comments": "", "author_explanations": ["quote1","quote2","quote3"] }. Paper: <<<INSERT PAPER TEXT>>> 

"""


# ======== GPT 调用函数 ========
def gpt5_inference(prompt, text, model="gpt-5", max_tokens=800):
    """
    Robust GPT-5 inference with fallback and retry
    """
    import re
    MAX_INPUT_CHARS = 48000  # GPT-5 安全输入上限（约 32k tokens）

    # ---- Step 1: 截断超长输入 ----
    if len(text) > MAX_INPUT_CHARS:
        print(f"⚠️ Input too long ({len(text)} chars). Truncating...")
        text = text[:MAX_INPUT_CHARS]

    full_prompt = f"Article:\n{text}\n\nTask:\n{prompt}"
    print(f"📏 Effective input length: {len(full_prompt)} chars")

    messages = [
        {"role": "system", "content": "You are an expert in alloy synthesis extraction."},
        {"role": "user", "content": full_prompt},
    ]

    # ---- Step 2: 尝试多次请求 ----
    for attempt in range(3):
        try:
            # 自动兼容新版 / 旧版参数
            kwargs = {}
            try:
                kwargs["max_output_tokens"] = max_tokens
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)
            except TypeError:
                kwargs = {"max_completion_tokens": max_tokens}
                response = client.chat.completions.create(model=model, messages=messages, **kwargs)

            result = response.choices[0].message.content.strip()

            if not result or result.strip() in ["...", "…"]:
                print(f"⚠️ Empty or truncated response on attempt {attempt+1}, retrying...")
                time.sleep(2)
                continue

            # ---- Step 3: 打印返回信息 ----
            print(f"🧩 GPT result length: {len(result)} chars")
            print(f"🧩 Preview: {result[:120]}...")
            return result

        except Exception as e:
            print(f"❌ GPT-5 attempt {attempt+1} failed: {e}")
            time.sleep(3)

    print("🚫 All retries failed or response empty.")
    return None


# ======== 单文件处理 ========
def process_single_file(file_path, output_dir):
    try:
        print(f"\n🧩 Processing {file_path}")
        text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 50:
            print(f"⚠️ {file_path.name} is empty or too short, skipping.")
            return {"id": Path(file_path).name, "method_output": "Empty input"}

        result = gpt5_inference(SYNTHESIS_PROMPT, text)
        if not result:
            print(f"⚠️ No valid result for {file_path}")
            return {"id": Path(file_path).name, "method_output": "Empty result"}

        output_file = Path(output_dir) / Path(file_path).name
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())

        print(f"✅ Saved {output_file.name}")
        return {"id": Path(file_path).name, "method_output": result}

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return {"id": Path(file_path).name, "method_output": f"Error: {e}"}


# ======== 主函数（多线程） ========
def process_files(input_directory, output_directory, max_threads=5):
    input_dir = Path(input_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted([f for f in input_dir.glob("*.txt")])
    total_files = len(txt_files)
    print(f"📚 Found {total_files} text files in {input_dir}")

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_file = {executor.submit(process_single_file, f, output_dir): f for f in txt_files}

        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ [{i}/{total_files}] {file_path.name} done.")
            except Exception as e:
                print(f"❌ [{i}/{total_files}] {file_path.name} failed: {e}")

    # 保存汇总 JSON
    summary_file = output_dir / "all_results.json"
    with open(summary_file, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)

    duration = time.time() - start_time
    print(f"\n🎯 Finished processing {total_files} files in {duration:.2f}s.")
    print(f"📄 Results saved to {summary_file}")


# ======== CLI ========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-threaded GPT-5 alloy synthesis extractor.")
    parser.add_argument("--input_directory", type=str, required=True, help="Directory containing .txt files")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save results")
    parser.add_argument("--threads", type=int, default=5, help="Number of parallel threads (default=5)")
    args = parser.parse_args()

    process_files(args.input_directory, args.output_directory, args.threads)
