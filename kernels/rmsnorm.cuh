#pragma once
// ---------------------------------------------------------------------------
// rmsnorm.cuh – RMSNorm kernel declarations
//
// Optimization progression (all variants preserved for benchmarking):
//   v0_naive       – one block, global atomics for sum-of-squares
//   v1_block       – one block per row, parallel block reduction
//   v2_warp        – warp-shuffle reduction (no __shared__ for reduction)
//   v3_vec         – vectorized load (float4 / half2) + warp shuffle
//   v4_fused       – fused residual add + normalize in one pass
//
// Active runtime uses v4_fused.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cstdint>

// One block per row of the hidden dimension.
// hidden_size must be a multiple of 32 for the warp-shuffle variants.

// v0 – naive reference
void rmsnorm_v0_naive(
    const __half* __restrict__ input,   // [batch, hidden_size]
    const __half* __restrict__ weight,  // [hidden_size]
    __half*       __restrict__ output,  // [batch, hidden_size]
    int batch, int hidden_size, float eps,
    cudaStream_t stream = nullptr);

// v1 – parallel block reduction
void rmsnorm_v1_block(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int batch, int hidden_size, float eps,
    cudaStream_t stream = nullptr);

// v2 – warp-shuffle reduction
void rmsnorm_v2_warp(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int batch, int hidden_size, float eps,
    cudaStream_t stream = nullptr);

// v3 – vectorized loads (half2) + warp shuffle
void rmsnorm_v3_vec(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int batch, int hidden_size, float eps,
    cudaStream_t stream = nullptr);

// v4 – fused residual add + normalize (two inputs, one output)
// output = rms_norm(input + residual) * weight;  residual is updated in-place.
void rmsnorm_v4_fused(
    const __half* __restrict__ input,    // [batch, hidden_size]
    __half*       __restrict__ residual, // [batch, hidden_size] – updated in-place
    const __half* __restrict__ weight,   // [hidden_size]
    __half*       __restrict__ output,   // [batch, hidden_size]
    int batch, int hidden_size, float eps,
    cudaStream_t stream = nullptr);
