#pragma once
// ---------------------------------------------------------------------------
// elementwise.cuh – SwiGLU, residual add, and misc fused elementwise ops
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>

// SwiGLU: out[i] = gate[i] * silu(gate[i]) * up[i]
// Both gate and up are [batch, intermediate_size] fp16.
// Output overwrites gate in-place.
void swiglu_inplace(
    __half* __restrict__ gate,          // [batch, intermediate_size] – modified in-place
    const __half* __restrict__ up,      // [batch, intermediate_size]
    int batch, int intermediate_size,
    cudaStream_t stream = nullptr);

// Simple residual add: a += b  (in-place)
void residual_add(
    __half* __restrict__ a,
    const __half* __restrict__ b,
    int n,
    cudaStream_t stream = nullptr);
