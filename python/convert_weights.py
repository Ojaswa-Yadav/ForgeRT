#!/usr/bin/env python3
"""
convert_weights.py – Download and prepare a Llama/Qwen-style model for ForgeRT.

Steps:
  1. Download model from HuggingFace Hub.
  2. Verify it is in safetensors format (convert from pytorch_model.bin if not).
  3. Print a summary of tensor names, shapes, and dtypes.
  4. Optionally merge shards into a single model.safetensors.

Usage:
    python convert_weights.py --model_id meta-llama/Llama-3.2-1B \
                              --output_dir ./models/llama-3.2-1b
    python convert_weights.py --model_id Qwen/Qwen2-0.5B \
                              --output_dir ./models/qwen2-0.5b

Notes:
  - Some models (Llama-3.2) require HF access tokens.
    Set: export HF_TOKEN=hf_...
  - ForgeRT expects tokenizer.json and config.json in the same directory.
"""
import argparse
import os
import json
import struct
from pathlib import Path


def download_from_hub(model_id: str, output_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install huggingface_hub")

    token = os.environ.get("HF_TOKEN")
    print(f"[convert] Downloading {model_id} → {output_dir} …")
    path = snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"[convert] Downloaded to {path}")
    return path


def parse_safetensors_header(path: str) -> dict:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
    header = json.loads(header_json)
    return {k: v for k, v in header.items() if k != "__metadata__"}


def summarise(output_dir: str):
    from pathlib import Path
    st_files = sorted(Path(output_dir).glob("*.safetensors"))
    if not st_files:
        print("[convert] No .safetensors files found.")
        return

    print(f"\n[convert] Found {len(st_files)} safetensors file(s):")
    total_params = 0
    for f in st_files:
        meta = parse_safetensors_header(str(f))
        print(f"\n  {f.name}:")
        for name, info in sorted(meta.items()):
            shape = info["shape"]
            dtype = info["dtype"]
            n = 1
            for s in shape: n *= s
            total_params += n
            print(f"    {name:60s}  {str(shape):25s}  {dtype}")
    print(f"\nTotal parameters: {total_params/1e9:.3f}B")


def convert_bin_to_safetensors(output_dir: str):
    """Convert pytorch_model.bin shards to safetensors if needed."""
    from pathlib import Path
    bin_files = sorted(Path(output_dir).glob("pytorch_model*.bin"))
    if not bin_files:
        return  # Nothing to convert

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError("pip install torch safetensors")

    print(f"[convert] Converting {len(bin_files)} .bin shard(s) to safetensors …")
    state_dict = {}
    for bf in bin_files:
        sd = torch.load(str(bf), map_location="cpu")
        state_dict.update(sd)

    out_path = Path(output_dir) / "model.safetensors"
    save_file(state_dict, str(out_path))
    print(f"[convert] Saved → {out_path}")

    # Clean up bin files
    for bf in bin_files:
        bf.unlink()
    print("[convert] Removed .bin files.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",   default=None,
                        help="HuggingFace model ID (e.g. Qwen/Qwen2-0.5B)")
    parser.add_argument("--output_dir", required=True,
                        help="Local directory for the model")
    parser.add_argument("--summarise",  action="store_true",
                        help="Only print tensor summary (don't download)")
    args = parser.parse_args()

    if not args.summarise:
        if not args.model_id:
            parser.error("--model_id required unless --summarise")
        download_from_hub(args.model_id, args.output_dir)
        convert_bin_to_safetensors(args.output_dir)

    summarise(args.output_dir)


if __name__ == "__main__":
    main()
