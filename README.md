# ForgeRT — CUDA-Native LLM Decode Engine

A single-GPU autoregressive transformer inference runtime in C++/CUDA.
No PyTorch on the critical inference path.

**Tested on:** Qwen2-0.5B · A40 (sm_86) · CUDA 12.8

---

## Results

| Model | GPU | Decode | Prefill |
|-------|-----|--------|---------|
| Qwen2-0.5B | A40 (45GB) | **323 tok/s** | 42 tok/s |

- 14/14 kernel correctness tests passing
- 6.4 GB VRAM for Qwen2-0.5B in FP16
- ~320 GB/s achieved memory bandwidth (~46% of A40 peak)

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Correct FP16 runtime (cuBLAS GEMMs, safetensors loader) | ✅ Done |
| 2 | Hand-optimized RMSNorm + RoPE kernels | ✅ Done |
| 3 | Flash-decode attention (warp-cooperative, online softmax) | ✅ Done |
| 4 | INT4/INT8 weight-only quantization | ⏳ |
| 5 | Custom FP16 GEMM (replace cuBLAS) | ⏳ |
| 6 | Paged KV cache | ⏳ stretch |
| 7 | Continuous batching | ⏳ stretch |

---

## Architecture

```
CLI (main.cpp)
    ↓
C++ inference runtime  (runtime/)
    model.cpp     – safetensors loader, BF16/F32→FP16 conversion, GPU pool
    executor.cpp  – transformer layer orchestration, cuBLAS GEMMs
    kv_cache.cpp  – contiguous KV cache [layers × 2 × seq_len × kv_heads × head_dim]
    main.cpp      – CLI, CUDA event timing, greedy decode loop
    ↓
CUDA kernels  (kernels/)
    rmsnorm.cu          – v0 naive → v4 fused residual+norm
    rope.cu             – v0 naive → v2 fused Q+K, precomputed sin/cos tables
    attention_decode.cu – v0 naive → v4 flash-decode (warp-cooperative)
    elementwise.cu      – SwiGLU, residual add, bias broadcast
    embed.cu            – embedding gather
    sampling.cu         – two-phase parallel argmax
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
ninja -j$(nproc)   # or: make -j$(nproc)
cd ..
```

### 3. Download a model

```bash
# Qwen2-0.5B (~1 GB FP16, recommended for quick start)
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-0.5B', local_dir='./models/qwen2-0.5b')
"
```

### 4. Get token IDs for your prompt

```bash
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./models/qwen2-0.5b')
print(tok.encode('The capital of France is'))
"
# [785, 6722, 315, 9625, 374]
```

### 5. Run

```bash
./build/forgert ./models/qwen2-0.5b \
    --tokens 785,6722,315,9625,374 \
    --max_tokens 50
```

### 6. Run kernel correctness tests

```bash
./build/test_kernels
# Expected: 14/14 tests passed
```

---

## Kernel Optimization Progression

### RMSNorm  (`kernels/rmsnorm.cu`)

| Variant | Technique |
|---------|-----------|
| v0_naive | Global atomics, serial accumulation |
| v1_block | Parallel block reduction in shared memory |
| v2_warp  | Warp-shuffle reduction (`__shfl_xor_sync`) |
| v3_vec   | Vectorized `half2` loads + warp shuffle |
| v4_fused | Fused residual add + normalize in one pass |

### RoPE  (`kernels/rope.cu`)

| Variant | Technique |
|---------|-----------|
| v0_naive | Scalar, recomputes sin/cos each call |
| v1_coal  | Coalesced loads, precomputed tables |
| v2_fused | Q and K rotated in one kernel launch |

### Decode Attention  (`kernels/attention_decode.cu`)

| Variant | Technique | Register spill |
|---------|-----------|----------------|
| v0_naive | Thread 0 serial over kv_len | 0 |
| v1_coal  | Threads split kv_len, two-pass softmax | 0 |
| v2_warp_red | Online softmax per warp, warp-merge | 512B stack |
| v3_vec   | Vectorized `half2` KV loads + online softmax | 512B stack |
| v4_flash | All 32 lanes cooperate per KV position, `acc[EPL]` per lane | **0** |

v4 design: warps split `kv_len`, lanes split `head_dim`. Each lane holds
`HEAD_DIM/32` accumulator elements — 4 floats for HD=128. No spill.
Cross-warp merge via shared memory. Confirmed by ptxas: `0 bytes stack frame`.

---

## Prefill vs Decode

| Dimension | Prefill | Decode |
|-----------|---------|--------|
| Query shape | `[T, H]` | `[1, H]` |
| Bottleneck | Compute (large GEMMs) | Memory bandwidth |
| Attention | `O(T²)` causal | Reads full KV cache |
| Roofline ceiling (A40, FP16) | ~300 TFLOP/s | ~696 GB/s |

At batch=1 decode, every token reads ~1 GB of weights (0.5B params × 2 bytes).
A40 peak bandwidth is 696 GB/s → ceiling ~700 tok/s. We hit ~323 tok/s (~46%).
The gap is mainly small-matrix cuBLAS underutilization + kernel launch overhead
across 24 layers. INT4 quantization would raise the ceiling by ~1.6–1.7x
(embedding and LM head stay FP16).

---

## Repository Layout

```
forgert/
├── CMakeLists.txt
├── runtime/
│   ├── include/
│   │   ├── types.h        – DeviceBuffer, ModelConfig, CUDA_CHECK
│   │   ├── model.h        – ModelWeights, LayerWeights (QKV biases included)
│   │   ├── kv_cache.h     – contiguous KV cache
│   │   └── executor.h     – Executor, ActivationBuffers
│   ├── model.cpp          – safetensors parser, optional bias loading
│   ├── executor.cpp       – layer orchestration, hgemm helper
│   ├── kv_cache.cpp
│   └── main.cpp
├── kernels/
│   ├── rmsnorm.cuh / .cu
│   ├── rope.cuh / .cu
│   ├── attention.cuh / attention_decode.cu
│   ├── elementwise.cuh / .cu  (SwiGLU, residual add, bias broadcast)
│   ├── embed.cuh / .cu
│   └── sampling.cu
├── benchmarks/
│   ├── bench_attention.cpp
│   ├── bench_gemm.cpp
│   └── roofline.py
├── tests/
│   └── test_kernels.cpp   – 14 correctness tests
├── python/
│   ├── convert_weights.py
│   ├── reference.py
│   ├── tokenizer.py
│   └── benchmark.py
├── scripts/
│   ├── setup_deps.sh
│   └── runpod_setup.sh    – one-shot cloud GPU setup
└── third_party/
    └── nlohmann/json.hpp  (downloaded by setup_deps.sh)
```

---

## Known Limitations

- Tokenizer is a stub — pass `--tokens` with pre-encoded IDs from Python
- Single safetensors shard only (no multi-shard checkpoint support yet)
- Prefill attention is single-threaded (correctness baseline, not optimized)
- No beam search, temperature sampling, or top-p yet (greedy only)
