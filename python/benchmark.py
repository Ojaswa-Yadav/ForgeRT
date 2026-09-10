#!/usr/bin/env python3
"""
benchmark.py – End-to-end latency benchmark comparing ForgeRT vs HuggingFace.

Usage:
    python python/benchmark.py \
        --model     ./models/qwen2-0.5b \
        --forgert   ./build/forgert \
        --prompt    "Explain quantum entanglement in simple terms" \
        --max_tokens 100

Outputs a CSV-compatible table:
    backend, prefill_ms, avg_decode_ms, tokens_s
"""
import argparse
import subprocess
import time
import json
import os
import sys


def benchmark_hf(model_dir: str, prompt: str, max_tokens: int) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()

    inputs = tok(prompt, return_tensors="pt").to("cuda")
    n_input = inputs["input_ids"].shape[1]

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    n_generated = out.shape[1] - n_input
    total_ms     = (t1 - t0) * 1000
    avg_ms_tok   = total_ms / max(n_generated, 1)

    return {
        "backend":        "HuggingFace FP16",
        "total_ms":       round(total_ms, 2),
        "avg_decode_ms":  round(avg_ms_tok, 2),
        "tokens_s":       round(1000 / avg_ms_tok, 1),
        "n_generated":    n_generated,
    }


def benchmark_forgert(forgert_bin: str, model_dir: str,
                       prompt: str, max_tokens: int) -> dict:
    cmd = [forgert_bin, model_dir,
           "--prompt", prompt,
           "--max_tokens", str(max_tokens)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[Error] forgert failed:", result.stderr)
        return {}

    lines = result.stdout.splitlines()
    metrics = {}
    for line in lines:
        if "Prefill latency:" in line:
            parts = line.split()
            # "[ForgeRT] Prefill latency: 12.3 ms ..."
            metrics["prefill_ms"] = float(parts[parts.index("latency:") + 1])
        if "Decode:" in line and "avg" in line:
            parts = line.split()
            # "[ForgeRT] Decode: N tokens  avg X ms/tok  (Y tokens/s)"
            avg_idx = parts.index("avg") + 1
            metrics["avg_decode_ms"] = float(parts[avg_idx])
            tok_idx = parts.index("tokens/s)") - 1
            metrics["tokens_s"] = float(parts[tok_idx].lstrip("("))

    metrics["backend"] = "ForgeRT (FP16)"
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True)
    parser.add_argument("--forgert",    default="./build/forgert")
    parser.add_argument("--prompt",     default="The meaning of life is")
    parser.add_argument("--max_tokens", type=int, default=50)
    args = parser.parse_args()

    results = []

    print("=== ForgeRT vs HuggingFace Benchmark ===\n")
    print(f"Model:      {args.model}")
    print(f"Prompt:     '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}\n")

    # ForgeRT
    print("Running ForgeRT …")
    r = benchmark_forgert(args.forgert, args.model, args.prompt, args.max_tokens)
    if r:
        results.append(r)
        print(f"  Prefill:      {r.get('prefill_ms', '?')} ms")
        print(f"  Avg decode:   {r.get('avg_decode_ms', '?')} ms/tok")
        print(f"  Throughput:   {r.get('tokens_s', '?')} tokens/s")

    # HuggingFace
    print("\nRunning HuggingFace …")
    try:
        r = benchmark_hf(args.model, args.prompt, args.max_tokens)
        results.append(r)
        print(f"  Total time:   {r['total_ms']} ms")
        print(f"  Avg decode:   {r['avg_decode_ms']} ms/tok")
        print(f"  Throughput:   {r['tokens_s']} tokens/s")
    except ImportError:
        print("  (transformers not installed — skipping HF baseline)")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Backend':<25} {'avg_decode(ms)':>16} {'tokens/s':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r.get('backend','?'):<25} {r.get('avg_decode_ms','?'):>16} {r.get('tokens_s','?'):>12}")


if __name__ == "__main__":
    main()
