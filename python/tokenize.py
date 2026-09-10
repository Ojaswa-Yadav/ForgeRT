#!/usr/bin/env python3
"""
tokenize.py – Tokenize a prompt using HuggingFace and print token IDs.

Usage:
    python python/tokenize.py --model <dir> --text "Hello world"

Output (to stdout):
    <bos_id> <tok1> <tok2> ...

Pipe into forgert for accurate tokenization:
    forgert <model_dir> --prompt "$(python python/tokenize.py --model <dir> --text '...')"
"""
import argparse
import sys
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text",  required=True)
    parser.add_argument("--decode", action="store_true",
                        help="Decode token IDs from stdin instead")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    if args.decode:
        ids = list(map(int, sys.stdin.read().split()))
        print(tok.decode(ids, skip_special_tokens=False))
    else:
        ids = tok.encode(args.text, add_special_tokens=True)
        print(" ".join(map(str, ids)))


if __name__ == "__main__":
    main()
