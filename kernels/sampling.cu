// ---------------------------------------------------------------------------
// sampling.cu  –  Greedy sampling (argmax over logits)
//
// Input: fp32 logits of shape [vocab_size]  (already on GPU).
// Output: single int32 token id.
//
// Two-phase parallel reduction:
//   Phase 1: each block reduces its chunk to a (max_val, max_idx) pair.
//   Phase 2: a single block reduces the block-level results.
// ---------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <float.h>
#include <cstdint>

struct MaxPair { float val; int32_t idx; };

__device__ __forceinline__ MaxPair max_pair(MaxPair a, MaxPair b) {
    return (a.val >= b.val) ? a : b;
}

__device__ __forceinline__ MaxPair warp_reduce_max(MaxPair v) {
    for (int off = 16; off > 0; off >>= 1) {
        MaxPair other;
        other.val = __shfl_xor_sync(0xFFFFFFFF, v.val, off);
        other.idx = __shfl_xor_sync(0xFFFFFFFF, v.idx, off);
        v = max_pair(v, other);
    }
    return v;
}

// Phase 1: reduce [vocab_size] to [num_blocks] MaxPairs
__global__ void argmax_phase1(
    const float* __restrict__ logits,
    MaxPair*     __restrict__ block_results,
    int vocab_size)
{
    extern __shared__ MaxPair sh[];

    int tid  = threadIdx.x;
    int base = blockIdx.x * blockDim.x;

    MaxPair local = { -FLT_MAX, -1 };
    for (int i = base + tid; i < vocab_size; i += gridDim.x * blockDim.x) {
        if (logits[i] > local.val) { local.val = logits[i]; local.idx = i; }
    }

    // Warp reduction
    local = warp_reduce_max(local);
    int lane = tid & 31;
    int wid  = tid >> 5;
    if (lane == 0) sh[wid] = local;
    __syncthreads();

    // Final warp
    int num_warps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        MaxPair v = (lane < num_warps) ? sh[lane] : MaxPair{-FLT_MAX, -1};
        v = warp_reduce_max(v);
        if (lane == 0) block_results[blockIdx.x] = v;
    }
}

// Phase 2: reduce block_results to a single int
__global__ void argmax_phase2(
    const MaxPair* __restrict__ block_results,
    int32_t*       __restrict__ out_idx,
    int num_blocks)
{
    extern __shared__ MaxPair sh2[];
    int tid = threadIdx.x;

    MaxPair local = { -FLT_MAX, -1 };
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local = max_pair(local, block_results[i]);
    }
    local = warp_reduce_max(local);
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) sh2[wid] = local;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    if (wid == 0) {
        MaxPair v = (lane < num_warps) ? sh2[lane] : MaxPair{-FLT_MAX, -1};
        v = warp_reduce_max(v);
        if (lane == 0) *out_idx = v.idx;
    }
}

// Host-callable entry point.
// block_buf: device buffer of [num_blocks] MaxPair (provided by caller).
// out_idx:   device int32 (provided by caller).
void greedy_sample(
    const float* logits,
    MaxPair*     block_buf,
    int32_t*     out_idx,
    int          vocab_size,
    cudaStream_t stream)
{
    int threads   = 256;
    int num_warps = threads / 32;
    int num_blocks = (vocab_size + threads - 1) / threads;
    num_blocks = min(num_blocks, 1024);

    argmax_phase1<<<num_blocks, threads, num_warps * sizeof(MaxPair), stream>>>(
        logits, block_buf, vocab_size);

    int warps2 = (num_blocks + 31) / 32;
    // Use a single block for phase 2
    argmax_phase2<<<1, min(num_blocks, 256),
                    ((min(num_blocks, 256) + 31) / 32) * sizeof(MaxPair), stream>>>(
        block_buf, out_idx, num_blocks);
}
