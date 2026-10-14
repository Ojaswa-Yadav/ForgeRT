#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# runpod_setup.sh — One-shot setup for ForgeRT on a RunPod (or Lambda Labs)
# GPU instance.  Run as root (the default on RunPod).
#
# Recommended pod:  RunPod "RTX 3090" or "A100 SXM" template
#   Image:  runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
#   (any CUDA-devel image ≥ 11.8 works)
#
# Usage:
#   1. SSH into the pod.
#   2. git clone <your-repo> forgert && cd forgert
#   3. bash scripts/runpod_setup.sh
#   4. Follow the printed "Next steps".
# ---------------------------------------------------------------------------
set -euo pipefail

echo "======================================================"
echo "  ForgeRT — RunPod / Lambda Labs setup"
echo "======================================================"

# ── 0. Detect GPU arch ─────────────────────────────────────────────────────
GPU_ARCH=$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode()
cc = out.strip().split('\n')[0].replace('.','')
print(cc)
" 2>/dev/null || echo "86")
echo "[setup] Detected compute capability: sm_${GPU_ARCH}"

# ── 1. System packages ─────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl wget \
    python3 python3-pip python3-venv
echo "[setup] System packages OK"

# ── 2. Python deps (for weight conversion + reference check) ───────────────
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet \
    torch==2.2.0 transformers accelerate safetensors huggingface_hub \
    numpy matplotlib
echo "[setup] Python deps OK"

# ── 3. Header-only C++ deps ────────────────────────────────────────────────
bash "$(dirname "$0")/setup_deps.sh"

# ── 4. Build ───────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${GPU_ARCH}" \
    -G Ninja

ninja -j"$(nproc)"
echo "[setup] Build complete: $(ls "$BUILD_DIR"/{forgert,test_kernels,bench_attention,bench_gemm} 2>/dev/null | tr '\n' ' ')"

# ── 5. Kernel correctness tests ────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  Running kernel correctness tests …"
echo "======================================================"
"$BUILD_DIR/test_kernels"

# ── 6. Print next steps ────────────────────────────────────────────────────
cat <<'NEXTSTEPS'

======================================================
  Build passed.  Next steps:
======================================================

1. Download Qwen2-0.5B weights (~1 GB):
     python3 python/convert_weights.py \
         --model_id Qwen/Qwen2-0.5B \
         --output_dir ./models/qwen2-0.5b

2. Generate reference output (PyTorch):
     python3 python/reference.py \
         --model_id Qwen/Qwen2-0.5B \
         --prompt "The capital of France is" \
         --max_tokens 20

3. Run ForgeRT inference:
     ./build/forgert ./models/qwen2-0.5b \
         --prompt "The capital of France is" \
         --max_tokens 20

4. Compare latency vs HuggingFace:
     python3 python/benchmark.py \
         --model_id Qwen/Qwen2-0.5B \
         --forgert_bin ./build/forgert \
         --model_dir ./models/qwen2-0.5b

5. (Optional) Roofline plot:
     python3 benchmarks/roofline.py

For Lambda Labs: same steps, just SSH into the instance first.
NEXTSTEPS
