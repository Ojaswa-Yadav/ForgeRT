// ---------------------------------------------------------------------------
// elementwise.cu  –  SwiGLU and residual add
// ---------------------------------------------------------------------------
#include "elementwise.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
__device__ __forceinline__ float silu(float x) {
    return x / (1.f + expf(-x));
}

// SwiGLU: out[i] = silu(gate[i]) * up[i]
// Both tensors are [batch, intermediate_size] in fp16.
// gate is overwritten with the result.
__global__ void swiglu_kernel(
    __half* __restrict__       gate,
    const __half* __restrict__ up,
    int n)   // total elements = batch * intermediate_size
{
    // Use half2 for vectorized loads (2 elements per thread per iteration)
    int idx2 = (blockIdx.x * blockDim.x + threadIdx.x);
    int total2 = n / 2;

    if (idx2 < total2) {
        __half2* g2 = reinterpret_cast<__half2*>(gate);
        const __half2* u2 = reinterpret_cast<const __half2*>(up);

        __half2 gv = g2[idx2];
        __half2 uv = u2[idx2];

        float g0 = silu(__half2float(__low2half(gv)));
        float g1 = silu(__half2float(__high2half(gv)));
        float u0 = __half2float(__low2half(uv));
        float u1 = __half2float(__high2half(uv));

        g2[idx2] = __halves2half2(__float2half(g0 * u0), __float2half(g1 * u1));
    }
}

void swiglu_inplace(
    __half* gate, const __half* up,
    int batch, int intermediate_size, cudaStream_t stream)
{
    int n = batch * intermediate_size;
    int threads = 256;
    int blocks  = (n / 2 + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, n);
}

// ---------------------------------------------------------------------------
// Residual add: a += b  (both are n fp16 elements)
// ---------------------------------------------------------------------------
__global__ void residual_add_kernel(
    __half* __restrict__       a,
    const __half* __restrict__ b,
    int n)
{
    int idx2   = blockIdx.x * blockDim.x + threadIdx.x;
    int total2 = n / 2;
    if (idx2 < total2) {
        __half2* a2       = reinterpret_cast<__half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        a2[idx2] = __hadd2(a2[idx2], b2[idx2]);
    }
}

void residual_add(
    __half* a, const __half* b, int n, cudaStream_t stream)
{
    int threads = 256;
    int blocks  = (n / 2 + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, n);
}
