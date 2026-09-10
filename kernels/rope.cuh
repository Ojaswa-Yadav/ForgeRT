#pragma once
// ---------------------------------------------------------------------------
// rope.cuh – Rotary Position Embedding (RoPE) kernel declarations
//
// Optimization progression:
//   v0_naive  – straightforward, scalar sin/cos per element pair
//   v1_coal   – coalesced access pattern, thread = one (cos,sin) pair
//   v2_fused  – fuse Q and K rotations in one kernel launch
//
// All variants apply RoPE in-place on the Q or K tensor.
// Input shape: [seq_len, num_heads, head_dim]  (head_dim must be even)
// The sin/cos tables are precomputed on the GPU at model load time.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cstdint>

// Precompute sin/cos table on device.
// Returns newly allocated float2 arrays (cos_table, sin_table) of shape
// [max_seq_len, head_dim/2].  Caller owns the memory.
void rope_precompute_tables(
    float** cos_out,     // [max_seq_len, head_dim/2]  device ptr
    float** sin_out,     // [max_seq_len, head_dim/2]  device ptr
    int max_seq_len, int head_dim,
    float base, float scale);

// v0 – naive per-element scalar
void rope_v0_naive(
    __half* __restrict__ q,             // [seq_len, num_heads,    head_dim]
    __half* __restrict__ k,             // [seq_len, num_kv_heads, head_dim]
    const float* __restrict__ cos_tab,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset,                     // starting position in sequence
    cudaStream_t stream = nullptr);

// v1 – coalesced access: one thread handles one (x_r, x_i) complex pair
void rope_v1_coal(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset,
    cudaStream_t stream = nullptr);

// v2 – fused Q+K in one kernel (avoids two launches)
void rope_v2_fused(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset,
    cudaStream_t stream = nullptr);
