#pragma once
// ---------------------------------------------------------------------------
// attention.cuh – KV-cache decode attention kernel declarations
//
// This is the centrepiece of ForgeRT.  Autoregressive decode presents a
// fundamentally memory-bandwidth-bound problem:
//   • Query shape:  [1, num_heads, head_dim]
//   • KV cache:     [kv_len, num_kv_heads, head_dim]  – grows every step
//
// Optimization progression (all variants preserved):
//
//   v0_naive      – one thread per KV position, sequential heads
//   v1_coal       – thread mapping reorganised for coalesced KV loads
//   v2_warp_red   – warp-level online softmax + weighted sum reduction
//   v3_vec        – vectorized half2/float4 loads from KV cache
//   v4_fused      – fuse QK^T, softmax, AV into one kernel (flash-decode style)
//
// Prefill path (seq_len > 1) uses a separate causal attention:
//   prefill_naive – simple tiled O(T^2) causal attention for correctness
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Decode attention  (seq_len == 1 or very small, kv_len can be large)
// ---------------------------------------------------------------------------

// v0 – naive: each thread handles one KV position, atomic accumulation
void decode_attn_v0_naive(
    const __half* __restrict__ q,          // [1, num_heads, head_dim]
    const __half* __restrict__ k_cache,    // [kv_len, num_kv_heads, head_dim]
    const __half* __restrict__ v_cache,    // [kv_len, num_kv_heads, head_dim]
    __half*       __restrict__ out,        // [1, num_heads, head_dim]
    int num_heads, int num_kv_heads, int head_dim, int kv_len,
    cudaStream_t stream = nullptr);

// v1 – coalesced KV loads: each warp handles one head, threads spread over kv_len
void decode_attn_v1_coal(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len,
    cudaStream_t stream = nullptr);

// v2 – warp-level online softmax + reduction
void decode_attn_v2_warp_red(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len,
    cudaStream_t stream = nullptr);

// v3 – vectorized loads (half2 for KV) + warp reduction
void decode_attn_v3_vec(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len,
    cudaStream_t stream = nullptr);

// v4 – fused flash-decode: one block per head, online softmax, no intermediate softmax buffer
void decode_attn_v4_flash(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half*       __restrict__ out,
    int num_heads, int num_kv_heads, int head_dim, int kv_len,
    cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// Prefill causal attention  (seq_len > 1, kv starts empty)
// ---------------------------------------------------------------------------
void prefill_attn_causal(
    const __half* __restrict__ q,          // [seq_len, num_heads,    head_dim]
    const __half* __restrict__ k,          // [seq_len, num_kv_heads, head_dim]
    const __half* __restrict__ v,          // [seq_len, num_kv_heads, head_dim]
    __half*       __restrict__ out,        // [seq_len, num_heads,    head_dim]
    int seq_len, int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream = nullptr);
