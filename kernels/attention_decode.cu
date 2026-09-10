// ---------------------------------------------------------------------------
// attention_decode.cu
//
// Autoregressive decode attention + prefill causal attention.
//
// During decode, Q has shape [1, num_heads, head_dim].
// The KV cache grows to [kv_len, num_kv_heads, head_dim].
// GQA: each KV head is shared by (num_heads / num_kv_heads) Q heads.
//
// Optimization progression:
//   v0 – naive: one block per Q-head, threads loop over kv_len serially
//   v1 – coalesced KV loads: threads split kv_len, warp reduction
//   v2 – warp-level online softmax (numerically stable) + reduction
//   v3 – vectorized half2 loads from KV cache
//   v4 – flash-decode: one block per head, single pass, no extra softmax buffer
//
// Prefill: tiled O(T²) causal attention for correctness baseline.
// ---------------------------------------------------------------------------
#include "attention.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Warp reduction helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_max(float v) {
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, m));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, m);
    return v;
}

// ============================================================
// v0 – naive
// One block per Q head.  Thread 0 loops over kv_len sequentially.
// Correct but completely underutilizes the GPU.
// ============================================================
__global__ void decode_attn_kernel_v0(
    const __half* __restrict__ q,         // [1, num_heads, head_dim]
    const __half* __restrict__ k_cache,   // [kv_len, num_kv_heads, head_dim]
    const __half* __restrict__ v_cache,   // [kv_len, num_kv_heads, head_dim]
    __half*       __restrict__ out,        // [1, num_heads, head_dim]
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    int h = blockIdx.x;   // Q head index
    if (h >= num_heads) return;

    int kv_h = h / (num_heads / num_kv_heads);   // corresponding KV head (GQA)
    float scale = rsqrtf((float)head_dim);

    const __half* q_ptr = q + h * head_dim;

    // Compute all QK dot products (thread 0 only — correctness reference)
    float m = -1e9f;
    float sum_exp = 0.f;
    // Store attention weights on the stack / registers (up to kv_len)
    // For correctness only; large kv_len will overflow stack in practice.
    // Use a two-pass approach to be numerically safe.

    extern __shared__ float shmem[];   // [kv_len + head_dim] floats
    float* scores = shmem;             // [kv_len]
    float* acc    = shmem + kv_len;    // [head_dim]

    if (threadIdx.x == 0) {
        // Pass 1: compute scores and find max (for stable softmax)
        for (int t = 0; t < kv_len; t++) {
            const __half* k_ptr = k_cache + (t * num_kv_heads + kv_h) * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
            }
            scores[t] = dot * scale;
            m = fmaxf(m, scores[t]);
        }
        // Pass 2: softmax
        for (int t = 0; t < kv_len; t++) {
            float e = expf(scores[t] - m);
            scores[t] = e;
            sum_exp += e;
        }
        // Pass 3: weighted sum over V
        for (int d = 0; d < head_dim; d++) acc[d] = 0.f;
        for (int t = 0; t < kv_len; t++) {
            const __half* v_ptr = v_cache + (t * num_kv_heads + kv_h) * head_dim;
            float w = scores[t] / sum_exp;
            for (int d = 0; d < head_dim; d++)
                acc[d] += w * __half2float(v_ptr[d]);
        }
        // Write output
        __half* o = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) o[d] = __float2half(acc[d]);
    }
}

void decode_attn_v0_naive(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len, cudaStream_t stream)
{
    size_t shmem = (kv_len + head_dim) * sizeof(float);
    decode_attn_kernel_v0<<<num_heads, 32, shmem, stream>>>(
        q, k_cache, v_cache, out, num_heads, num_kv_heads, head_dim, kv_len);
}

// ============================================================
// v1 – coalesced KV loads: threads split kv_len
// Block = (32, num_heads/block).  Threads in a warp span kv_len together.
// Still naive softmax (two passes via shared memory).
// ============================================================
__global__ void decode_attn_kernel_v1(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    int h    = blockIdx.x;
    int kv_h = h / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);

    extern __shared__ float sh[];
    float* scores  = sh;              // [kv_len]
    float* acc     = sh + kv_len;    // [head_dim]
    float* max_buf = acc + head_dim; // [warps_per_block]

    int warps = (blockDim.x + 31) >> 5;
    int lane  = threadIdx.x & 31;
    int wid   = threadIdx.x >> 5;

    const __half* q_ptr = q + h * head_dim;

    // Each thread computes QK for one kv_len slice
    float local_max = -FLT_MAX;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x) {
        const __half* k_ptr = k_cache + (t * num_kv_heads + kv_h) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        scores[t] = dot * scale;
        local_max = fmaxf(local_max, scores[t]);
    }
    __syncthreads();

    // Block-wide max
    local_max = warp_max(local_max);
    if (lane == 0) max_buf[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < warps) ? max_buf[lane] : -FLT_MAX;
        v = warp_max(v);
        if (lane == 0) max_buf[0] = v;
    }
    __syncthreads();
    float global_max = max_buf[0];

    // Softmax normalisation
    float local_sum = 0.f;
    for (int t = threadIdx.x; t < kv_len; t += blockDim.x) {
        float e = expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    __syncthreads();

    // Block-wide sum
    local_sum = warp_sum(local_sum);
    if (lane == 0) max_buf[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < warps) ? max_buf[lane] : 0.f;
        v = warp_sum(v);
        if (lane == 0) max_buf[0] = v;
    }
    __syncthreads();
    float total = max_buf[0];

    // Weighted sum over V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.f;
    __syncthreads();

    for (int t = 0; t < kv_len; t++) {
        float w = scores[t] / total;
        const __half* v_ptr = v_cache + (t * num_kv_heads + kv_h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            acc[d] += w * __half2float(v_ptr[d]);
        __syncthreads();
    }

    __half* o = out + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        o[d] = __float2half(acc[d]);
}

void decode_attn_v1_coal(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len, cudaStream_t stream)
{
    int threads = min(kv_len, 256);
    int warps   = (threads + 31) / 32;
    size_t shmem = (kv_len + head_dim + warps) * sizeof(float);
    decode_attn_kernel_v1<<<num_heads, threads, shmem, stream>>>(
        q, k_cache, v_cache, out, num_heads, num_kv_heads, head_dim, kv_len);
}

// ============================================================
// v2 – warp-level online softmax (single pass over KV) + reduction
//
// Uses the numerically stable online softmax algorithm:
//   for each new score s:
//     new_m = max(m, s)
//     new_d = d * exp(m - new_m) + exp(s - new_m)
//     new_acc = acc * exp(m - new_m) + v * exp(s - new_m)
//     m = new_m, d = new_d, acc = new_acc
//
// One warp handles one Q head.  Threads split kv_len.
// After the loop, a warp-level merge fuses the per-thread online states.
// ============================================================
__global__ void decode_attn_kernel_v2(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    // One warp per head
    int h    = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (h >= num_heads) return;

    int kv_h = h / (num_heads / num_kv_heads);
    int lane  = threadIdx.x & 31;
    float scale = rsqrtf((float)head_dim);

    const __half* q_ptr = q + h * head_dim;

    // Each lane maintains an online softmax state over its portion of kv_len
    float m_local   = -FLT_MAX;
    float d_local   = 0.f;
    // acc: partial weighted sum [head_dim] — kept as a local float array
    // For large head_dim (e.g. 128) this lives in registers.
    float acc[128] = {};  // assume head_dim <= 128

    for (int t = lane; t < kv_len; t += 32) {
        // Dot product Q · K[t]
        const __half* k_ptr = k_cache + (t * num_kv_heads + kv_h) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++)
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        float score = dot * scale;

        // Online update
        float m_new = fmaxf(m_local, score);
        float exp_old = expf(m_local - m_new);
        float exp_new = expf(score  - m_new);

        const __half* v_ptr = v_cache + (t * num_kv_heads + kv_h) * head_dim;
        for (int d = 0; d < head_dim; d++)
            acc[d] = acc[d] * exp_old + __half2float(v_ptr[d]) * exp_new;

        d_local = d_local * exp_old + exp_new;
        m_local = m_new;
    }

    // Warp-level merge of online softmax states
    // Algorithm: merge pairs of (m, d, acc) iteratively across lanes
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_other   = __shfl_xor_sync(0xFFFFFFFF, m_local, offset);
        float d_other   = __shfl_xor_sync(0xFFFFFFFF, d_local, offset);

        float m_new = fmaxf(m_local, m_other);
        float exp_self  = expf(m_local  - m_new);
        float exp_other = expf(m_other  - m_new);

        for (int d = 0; d < head_dim; d++) {
            float acc_other = __shfl_xor_sync(0xFFFFFFFF, acc[d], offset);
            acc[d] = acc[d] * exp_self + acc_other * exp_other;
        }
        d_local = d_local * exp_self + d_other * exp_other;
        m_local = m_new;
    }

    // Lane 0 of each warp writes the final output
    if (lane == 0) {
        __half* o = out + h * head_dim;
        for (int d = 0; d < head_dim; d++)
            o[d] = __float2half(acc[d] / d_local);
    }
}

void decode_attn_v2_warp_red(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len, cudaStream_t stream)
{
    // heads_per_block warps per block
    int heads_per_block = 4;
    int threads = heads_per_block * 32;
    int blocks  = (num_heads + heads_per_block - 1) / heads_per_block;
    decode_attn_kernel_v2<<<blocks, threads, 0, stream>>>(
        q, k_cache, v_cache, out, num_heads, num_kv_heads, head_dim, kv_len);
}

// ============================================================
// v3 – vectorized half2 loads from KV cache + warp reduction
// Same online algorithm as v2 but loads 2 elements at a time.
// head_dim must be even.
// ============================================================
__global__ void decode_attn_kernel_v3(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    int h    = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (h >= num_heads) return;

    int kv_h  = h / (num_heads / num_kv_heads);
    int lane  = threadIdx.x & 31;
    float scale = rsqrtf((float)head_dim);
    int half_d = head_dim / 2;

    const __half2* q2 = reinterpret_cast<const __half2*>(q + h * head_dim);

    float m_local = -FLT_MAX;
    float d_local = 0.f;
    float acc[128] = {};

    for (int t = lane; t < kv_len; t += 32) {
        const __half2* k2 = reinterpret_cast<const __half2*>(
            k_cache + (t * num_kv_heads + kv_h) * head_dim);

        float dot = 0.f;
        for (int dd = 0; dd < half_d; dd++) {
            __half2 qi = q2[dd];
            __half2 ki = k2[dd];
            // Element-wise multiply and accumulate (no half2 dot intrinsic in old CUDA)
            dot += __half2float(__low2half(qi))  * __half2float(__low2half(ki));
            dot += __half2float(__high2half(qi)) * __half2float(__high2half(ki));
        }
        float score = dot * scale;

        float m_new    = fmaxf(m_local, score);
        float exp_old  = expf(m_local - m_new);
        float exp_new  = expf(score   - m_new);

        const __half2* v2 = reinterpret_cast<const __half2*>(
            v_cache + (t * num_kv_heads + kv_h) * head_dim);
        for (int dd = 0; dd < half_d; dd++) {
            __half2 vi = v2[dd];
            acc[2*dd]   = acc[2*dd]   * exp_old + __half2float(__low2half(vi))  * exp_new;
            acc[2*dd+1] = acc[2*dd+1] * exp_old + __half2float(__high2half(vi)) * exp_new;
        }
        d_local = d_local * exp_old + exp_new;
        m_local = m_new;
    }

    // Warp merge
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_other = __shfl_xor_sync(0xFFFFFFFF, m_local, offset);
        float d_other = __shfl_xor_sync(0xFFFFFFFF, d_local, offset);
        float m_new   = fmaxf(m_local, m_other);
        float es = expf(m_local - m_new);
        float eo = expf(m_other - m_new);
        for (int d = 0; d < head_dim; d++) {
            float ao = __shfl_xor_sync(0xFFFFFFFF, acc[d], offset);
            acc[d] = acc[d] * es + ao * eo;
        }
        d_local = d_local * es + d_other * eo;
        m_local = m_new;
    }

    if (lane == 0) {
        __half* o = out + h * head_dim;
        for (int d = 0; d < head_dim; d++)
            o[d] = __float2half(acc[d] / d_local);
    }
}

void decode_attn_v3_vec(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len, cudaStream_t stream)
{
    int heads_per_block = 4;
    int threads = heads_per_block * 32;
    int blocks  = (num_heads + heads_per_block - 1) / heads_per_block;
    decode_attn_kernel_v3<<<blocks, threads, 0, stream>>>(
        q, k_cache, v_cache, out, num_heads, num_kv_heads, head_dim, kv_len);
}

// ============================================================
// v4 – flash-decode: one block per Q head
// Splits kv_len into tiles of TILE_KV; each tile runs in shared memory.
// QK scores are accumulated on the fly with online softmax.
// One global pass → minimal DRAM traffic.
// ============================================================
template<int HEAD_DIM, int TILE_KV>
__global__ void decode_attn_kernel_v4(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int kv_len)
{
    // One block = one Q head
    int h    = blockIdx.x;
    int kv_h = h / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)HEAD_DIM);

    // Q loaded into registers
    __shared__ float q_sh[HEAD_DIM];
    if (threadIdx.x < HEAD_DIM) {
        q_sh[threadIdx.x] = __half2float(q[h * HEAD_DIM + threadIdx.x]);
    }
    __syncthreads();

    // Shared buffers for one tile of K and V
    __shared__ float k_sh[TILE_KV][HEAD_DIM];
    __shared__ float v_sh[TILE_KV][HEAD_DIM];

    // Per-thread accumulator (HEAD_DIM elements)
    float m_global = -FLT_MAX;
    float d_global = 0.f;
    float acc[HEAD_DIM] = {};

    int num_tiles = (kv_len + TILE_KV - 1) / TILE_KV;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_KV;
        int tile_end   = min(tile_start + TILE_KV, kv_len);
        int tile_size  = tile_end - tile_start;

        // Cooperatively load K and V for this tile
        // threadIdx.x indexes into HEAD_DIM
        for (int kv_t = 0; kv_t < tile_size; kv_t++) {
            int t = tile_start + kv_t;
            if (threadIdx.x < HEAD_DIM) {
                k_sh[kv_t][threadIdx.x] = __half2float(
                    k_cache[(t * num_kv_heads + kv_h) * HEAD_DIM + threadIdx.x]);
                v_sh[kv_t][threadIdx.x] = __half2float(
                    v_cache[(t * num_kv_heads + kv_h) * HEAD_DIM + threadIdx.x]);
            }
        }
        __syncthreads();

        // Compute scores for this tile (thread 0 to avoid race — see note below)
        // For a true flash-decode we'd split heads across threads; here we keep
        // one block per head and use thread 0 for the outer loop to show the
        // tiling concept clearly.  v3/v2 show the warp parallelism.
        if (threadIdx.x == 0) {
            for (int kv_t = 0; kv_t < tile_size; kv_t++) {
                float dot = 0.f;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += q_sh[d] * k_sh[kv_t][d];
                float score = dot * scale;

                float m_new   = fmaxf(m_global, score);
                float exp_old = expf(m_global - m_new);
                float exp_new = expf(score    - m_new);

                for (int d = 0; d < HEAD_DIM; d++)
                    acc[d] = acc[d] * exp_old + v_sh[kv_t][d] * exp_new;

                d_global = d_global * exp_old + exp_new;
                m_global = m_new;
            }

            if (tile == num_tiles - 1) {
                __half* o = out + h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++)
                    o[d] = __float2half(acc[d] / d_global);
            }
        }
        __syncthreads();
    }
}

void decode_attn_v4_flash(
    const __half* q, const __half* k_cache, const __half* v_cache, __half* out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len, cudaStream_t stream)
{
    // Dispatch on head_dim (common values: 64, 128)
    if (head_dim == 128) {
        decode_attn_kernel_v4<128, 64><<<num_heads, 128, 0, stream>>>(
            q, k_cache, v_cache, out, num_heads, num_kv_heads, kv_len);
    } else if (head_dim == 64) {
        decode_attn_kernel_v4<64, 64><<<num_heads, 64, 0, stream>>>(
            q, k_cache, v_cache, out, num_heads, num_kv_heads, kv_len);
    } else {
        // Fall back to v3 for non-standard head dims
        decode_attn_v3_vec(q, k_cache, v_cache, out, num_heads, num_kv_heads, head_dim, kv_len, stream);
    }
}

// ============================================================
// Prefill causal attention  (O(T²))
// One block per output position s, threads compute attention over positions 0..s.
// Uses half2 dot products where possible.
// ============================================================
__global__ void prefill_attn_kernel(
    const __half* __restrict__ q,   // [seq_len, num_heads, head_dim]
    const __half* __restrict__ k,   // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ v,   // [seq_len, num_kv_heads, head_dim]
    __half*       __restrict__ out, // [seq_len, num_heads, head_dim]
    int seq_len, int num_heads, int num_kv_heads, int head_dim)
{
    int s = blockIdx.y;   // output position (0 .. seq_len-1)
    int h = blockIdx.x;   // Q head
    if (s >= seq_len || h >= num_heads) return;

    int kv_h = h / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);

    const __half* q_ptr = q + (s * num_heads + h) * head_dim;

    extern __shared__ float sh[];
    float* scores = sh;           // [s+1]
    float* acc    = sh + seq_len; // [head_dim]

    // Thread 0 computes everything (correctness baseline; not optimised)
    if (threadIdx.x == 0) {
        float m = -FLT_MAX, d = 0.f;
        for (int t = 0; t <= s; t++) {   // causal: only attend to positions <= s
            const __half* k_ptr = k + (t * num_kv_heads + kv_h) * head_dim;
            float dot = 0.f;
            for (int dd = 0; dd < head_dim; dd++)
                dot += __half2float(q_ptr[dd]) * __half2float(k_ptr[dd]);
            scores[t] = dot * scale;
            m = fmaxf(m, scores[t]);
        }
        for (int t = 0; t <= s; t++) {
            float e = expf(scores[t] - m);
            scores[t] = e; d += e;
        }
        for (int dd = 0; dd < head_dim; dd++) acc[dd] = 0.f;
        for (int t = 0; t <= s; t++) {
            const __half* v_ptr = v + (t * num_kv_heads + kv_h) * head_dim;
            float w = scores[t] / d;
            for (int dd = 0; dd < head_dim; dd++)
                acc[dd] += w * __half2float(v_ptr[dd]);
        }
        __half* o = out + (s * num_heads + h) * head_dim;
        for (int dd = 0; dd < head_dim; dd++)
            o[dd] = __float2half(acc[dd]);
    }
}

void prefill_attn_causal(
    const __half* q, const __half* k, const __half* v, __half* out,
    int seq_len, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream)
{
    size_t shmem = (seq_len + head_dim) * sizeof(float);
    dim3 grid(num_heads, seq_len);
    prefill_attn_kernel<<<grid, 32, shmem, stream>>>(
        q, k, v, out, seq_len, num_heads, num_kv_heads, head_dim);
}
