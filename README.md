# ForgeRT — CUDA-Native LLM Decode Engine

A single-GPU autoregressive transformer inference runtime in C++/CUDA.
No PyTorch on the critical inference path.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Correct FP16 runtime (cuBLAS GEMMs) | 🔄 In progress |
| 2 | Hand-optimized RMSNorm + RoPE kernels | ⏳ |
| 3 | Decode-optimized KV-cache attention | ⏳ |
| 4 | INT4/INT8 weight-only inference | ⏳ |
| 5 | Custom FP16 GEMM (vs cuBLAS) | ⏳ |
| 6 | Paged KV cache | ⏳ (stretch) |
| 7 | Continuous batching | ⏳ (stretch) |

---

## Architecture

```
CLI / Python test harness
    ↓
C++ inference runtime  (runtime/)
    model.cpp     – safetensors loader, GPU weight allocation
    executor.cpp  – transformer layer orchestration, cuBLAS GEMMs
    kv_cache.cpp  – contiguous KV cache
    main.cpp      – CLI, latency reporting
    ↓
CUDA kernels  (kernels/)
    rmsnorm.cu          – v0 naive → v4 fused residual+norm
    rope.cu             – v0 naive → v2 fused Q+K
    attention_decode.cu – v0 naive → v4 flash-decode
    elementwise.cu      – SwiGLU, residual add
    sampling.cu         – greedy argmax (two-phase reduction)
```

---

## Quick Start

### 1. Install dependencies

```bash
bash scripts/setup_deps.sh      # downloads nlohmann/json header
pip install huggingface_hub transformers safetensors
```

### 2. Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"  # set your GPU arch
make -j$(nproc)
cd ..
```

### 3. Download a model

```bash
# Qwen2-0.5B (0.5B params, ~1 GB FP16, fits on any CUDA GPU)
python python/convert_weights.py \
    --model_id Qwen/Qwen2-0.5B \
    --output_dir ./models/qwen2-0.5b

# Llama-3.2-1B (requires HuggingFace token for gated repo)
HF_TOKEN=hf_... python python/convert_weights.py \
    --model_id meta-llama/Llama-3.2-1B \
    --output_dir ./models/llama-3.2-1b
```

### 4. Run

```bash
./build/forgert ./models/qwen2-0.5b --prompt "The meaning of life is" --max_tokens 50
```

### 5. Validate against PyTorch

```bash
python python/reference.py \
    --model ./models/qwen2-0.5b \
    --prompt "The meaning of life is" \
    --max_tokens 5 \
    --output_logits /tmp/ref_logits.npy
```

### 6. Run kernel tests

```bash
./build/test_kernels
```

### 7. Run benchmarks

```bash
./build/bench_attention         # attention kernel sweep
./build/bench_gemm              # GEMM shapes (prefill vs decode)
python benchmarks/roofline.py --gpu 4090 --output roofline.png
```

---

## Kernel Optimization Progression

### RMSNorm  (`kernels/rmsnorm.cu`)

| Variant | Technique |
|---------|-----------|
| v0_naive | Global atomics, single thread accumulates |
| v1_block | Parallel block reduction in shared memory |
| v2_warp  | Warp-shuffle reduction (no sync for reduction) |
| v3_vec   | Vectorized `half2` loads + warp shuffle |
| v4_fused | Fused residual add + normalize in one pass |

### RoPE  (`kernels/rope.cu`)

| Variant | Technique |
|---------|-----------|
| v0_naive | Scalar, recomputes sin/cos per thread |
| v1_coal  | Coalesced: one thread per complex pair |
| v2_fused | Q and K in one kernel launch |

### Decode Attention  (`kernels/attention_decode.cu`)

| Variant | Technique |
|---------|-----------|
| v0_naive | Serial over kv_len, thread 0 only |
| v1_coal  | Threads split kv_len; two-pass softmax |
| v2_warp_red | Online softmax per warp, warp-merge |
| v3_vec   | Vectorized `half2` KV loads + online softmax |
| v4_flash | Flash-decode: tiled, single-pass, no softmax buffer |

---

## Prefill vs Decode Analysis

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| Query shape | `[T, H]` (T >> 1) | `[1, H]` |
| Matrix ops | Large GEMMs (compute-bound) | GEMV-like (memory-bound) |
| Attention | Compute-heavy `O(T²)` | KV-read / bandwidth-heavy |
| KV cache | Written | Read (entire cache) |
| Optimization | Compute utilization, Tensor Cores | Latency, memory bandwidth |

Use `benchmarks/roofline.py` to plot arithmetic intensity vs achieved throughput
and identify which kernels are memory- vs compute-bound.

---

## Repository Layout

```
forgert/
├── CMakeLists.txt
├── runtime/
│   ├── include/
│   │   ├── types.h        – DeviceBuffer, ModelConfig, CUDA_CHECK
│   │   ├── model.h        – ModelWeights, LayerWeights, safetensors parser
│   │   ├── kv_cache.h     – contiguous KV cache
│   │   └── executor.h     – Executor, ActivationBuffers
│   ├── model.cpp
│   ├── executor.cpp
│   ├── kv_cache.cpp
│   └── main.cpp
├── kernels/
│   ├── rmsnorm.cuh / .cu
│   ├── rope.cuh / .cu
│   ├── attention.cuh / attention_decode.cu
│   ├── elementwise.cuh / .cu
│   └── sampling.cu
├── benchmarks/
│   ├── bench_attention.cpp
│   ├── bench_gemm.cpp
│   └── roofline.py
├── tests/
│   └── test_kernels.cpp
├── python/
│   ├── convert_weights.py
│   ├── reference.py
│   ├── tokenize.py
│   └── benchmark.py
├── scripts/
│   └── setup_deps.sh
└── third_party/
    └── nlohmann/json.hpp  (downloaded by setup_deps.sh)
```
