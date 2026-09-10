#!/usr/bin/env python3
"""
reference.py – PyTorch reference implementation for logit validation.

Usage:
    python reference.py --model <hf_model_dir> --prompt "Hello world" \
                        --max_tokens 5 --output_logits logits.npy

Outputs:
  - Text to stdout (greedy decode)
  - Logits for each generated token as a numpy array

The ForgeRT C++ runtime should produce logits within atol=1e-1 of these
values (FP16 vs FP32 accumulation + kernel differences cause small gaps).
"""
import argparse
import json
import numpy as np
import torch


def load_model_and_tokenizer(model_dir: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    return tok, model


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    all_logits = []
    generated = input_ids.clone()

    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True)
        logits = out.logits[:, -1, :]          # [1, vocab]
        all_logits.append(logits.float().cpu().numpy())
        past = out.past_key_values

        next_tok = logits.argmax(-1)
        generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=-1)

        # Decode
        for _ in range(max_new_tokens - 1):
            out = model(next_tok.unsqueeze(0), past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            all_logits.append(logits.float().cpu().numpy())
            past = out.past_key_values
            next_tok = logits.argmax(-1)
            generated = torch.cat([generated, next_tok.unsqueeze(0)], dim=-1)

            if next_tok.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text, np.stack(all_logits, axis=0)   # [steps, 1, vocab]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True)
    parser.add_argument("--prompt",      default="The meaning of life is")
    parser.add_argument("--max_tokens",  type=int, default=5)
    parser.add_argument("--output_logits", default=None,
                        help="Save logits to this .npy file for comparison")
    args = parser.parse_args()

    print(f"[reference] Loading model from {args.model} …")
    tokenizer, model = load_model_and_tokenizer(args.model)

    print(f"[reference] Generating (prompt: '{args.prompt}') …")
    text, logits = greedy_generate(model, tokenizer, args.prompt, args.max_tokens)

    print(f"\n--- PyTorch output ---\n{text}\n")
    print(f"Logits shape: {logits.shape}")
    print(f"First token top-5: {logits[0, 0].argsort()[-5:][::-1].tolist()}")

    if args.output_logits:
        np.save(args.output_logits, logits)
        print(f"Saved logits to {args.output_logits}")


if __name__ == "__main__":
    main()
