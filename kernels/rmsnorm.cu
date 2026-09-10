// ---------------------------------------------------------------------------
// rmsnorm.cu
//
// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
//
// Optimization progression:
//   v0 – one thread accumulates the whole row (correctness reference)
//   v1 – parallel block reduction using shared memory
//   v2 – warp-shuffle reduction (eliminates most shared memory)
//   v3 – vectorized half2 loads + warp shuffle
//   v4 – fused residual add + normalize in a single pass
// ---------------------------------------------------------------------------
#include "rmsnorm.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// ============================================================
// v0 – naive: single thread computes the whole row sum-of-squares
// ============================================================
__global__ void rmsnorm_kernel_v0(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int hidden_size, float eps)
{
    int row = blockIdx.x;   // one block = one row
    const __half* x = input  + row * hidden_size;
    __half*       y = output + row * hidden_size;

    // Sum of squares (thread 0 only — purely for correctness reference)
    float ss = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        ss += xi * xi;
    }
    // Accumulate across block via shared memory barrier (simple but slow)
    __shared__ float shared_ss;
    if (threadIdx.x == 0) shared_ss = 0.f;
    __syncthreads();
    atomicAdd(&shared_ss, ss);
    __syncthreads();

    float rms_inv = rsqrtf(shared_ss / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        float wi = __half2float(weight[i]);
        y[i] = __float2half(xi * rms_inv * wi);
    }
}

void rmsnorm_v0_naive(
    const __half* input, const __half* weight, __half* output,
    int batch, int hidden_size, float eps, cudaStream_t stream)
{
    dim3 grid(batch);
    dim3 block(min(hidden_size, 256));
    rmsnorm_kernel_v0<<<grid, block, 0, stream>>>(input, weight, output, hidden_size, eps);
}

// ============================================================
// v1 – parallel block reduction using shared memory
// ============================================================
__global__ void rmsnorm_kernel_v1(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int hidden_size, float eps)
{
    extern __shared__ float shmem[];   // [blockDim.x]

    int row = blockIdx.x;
    const __half* x = input  + row * hidden_size;
    __half*       y = output + row * hidden_size;

    // Each thread accumulates its partial sum of squares
    float partial = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        partial += xi * xi;
    }
    shmem[threadIdx.x] = partial;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shmem[threadIdx.x] += shmem[threadIdx.x + stride];
        __syncthreads();
    }

    float rms_inv = rsqrtf(shmem[0] / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        float wi = __half2float(weight[i]);
        y[i] = __float2half(xi * rms_inv * wi);
    }
}

void rmsnorm_v1_block(
    const __half* input, const __half* weight, __half* output,
    int batch, int hidden_size, float eps, cudaStream_t stream)
{
    int threads = min(hidden_size, 512);
    // Round up to next power-of-2 for clean tree reduction
    while (threads & (threads - 1)) threads++;
    threads = min(threads, 1024);
    size_t shmem = threads * sizeof(float);
    rmsnorm_kernel_v1<<<batch, threads, shmem, stream>>>(input, weight, output, hidden_size, eps);
}

// ============================================================
// v2 – warp-shuffle reduction (no shared memory needed for reduction)
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Full-warp reduction using __shfl_xor_sync
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void rmsnorm_kernel_v2(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int hidden_size, float eps)
{
    // One warp per row (hidden_size <= 1024 assumed; padded launches for larger)
    // For hidden_size > 32 we use multiple warps per row.
    extern __shared__ float warp_sums[];   // [warps_per_block]

    int row  = blockIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    const __half* x = input  + row * hidden_size;
    __half*       y = output + row * hidden_size;

    float partial = 0.f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        partial += xi * xi;
    }

    // Warp-level reduction
    partial = warp_reduce_sum(partial);

    // First thread in each warp writes to shared memory
    if (lane == 0) warp_sums[wid] = partial;
    __syncthreads();

    // Final reduction: first warp reduces warp_sums
    if (wid == 0) {
        float val = (lane < num_warps) ? warp_sums[lane] : 0.f;
        val = warp_reduce_sum(val);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(warp_sums[0] / hidden_size + eps);

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float xi = __half2float(x[i]);
        float wi = __half2float(weight[i]);
        y[i] = __float2half(xi * rms_inv * wi);
    }
}

void rmsnorm_v2_warp(
    const __half* input, const __half* weight, __half* output,
    int batch, int hidden_size, float eps, cudaStream_t stream)
{
    int threads = min((hidden_size + 31) / 32 * 32, 1024);
    int num_warps = threads / 32;
    size_t shmem = num_warps * sizeof(float);
    rmsnorm_kernel_v2<<<batch, threads, shmem, stream>>>(input, weight, output, hidden_size, eps);
}

// ============================================================
// v3 – vectorized half2 loads + warp shuffle
// ============================================================
__global__ void rmsnorm_kernel_v3(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int hidden_size, float eps)
{
    extern __shared__ float warp_sums_v3[];

    int row = blockIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    const __half2* x2 = reinterpret_cast<const __half2*>(input  + row * hidden_size);
    const __half2* w2 = reinterpret_cast<const __half2*>(weight);
    __half2*       y2 = reinterpret_cast<__half2*>(output + row * hidden_size);

    int half_size = hidden_size / 2;  // assumes hidden_size even

    // Accumulate sum of squares using half2 arithmetic
    float partial = 0.f;
    for (int i = threadIdx.x; i < half_size; i += blockDim.x) {
        __half2 v = x2[i];
        float a = __half2float(__low2half(v));
        float b = __half2float(__high2half(v));
        partial += a * a + b * b;
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) warp_sums_v3[wid] = partial;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < num_warps) ? warp_sums_v3[lane] : 0.f;
        val = warp_reduce_sum(val);
        if (lane == 0) warp_sums_v3[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(warp_sums_v3[0] / hidden_size + eps);
    __half2 scale2 = __float2half2_rn(rms_inv);

    for (int i = threadIdx.x; i < half_size; i += blockDim.x) {
        __half2 v = x2[i];
        __half2 w = w2[i];
        // scale and weight: element-wise multiply
        __half2 r = __hmul2(__hmul2(v, scale2), w);
        y2[i] = r;
    }
}

void rmsnorm_v3_vec(
    const __half* input, const __half* weight, __half* output,
    int batch, int hidden_size, float eps, cudaStream_t stream)
{
    int threads = min((hidden_size / 2 + 31) / 32 * 32, 1024);
    int num_warps = threads / 32;
    size_t shmem = num_warps * sizeof(float);
    rmsnorm_kernel_v3<<<batch, threads, shmem, stream>>>(input, weight, output, hidden_size, eps);
}

// ============================================================
// v4 – fused residual add + normalize
// residual = input + residual   (residual updated in place)
// output   = rms_norm(residual) * weight
// Two global memory passes → one pass.  Critical for decode latency.
// ============================================================
__global__ void rmsnorm_kernel_v4_fused(
    const __half* __restrict__ input,
    __half*       __restrict__ residual,
    const __half* __restrict__ weight,
    __half*       __restrict__ output,
    int hidden_size, float eps)
{
    extern __shared__ float warp_sums_v4[];

    int row = blockIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    const __half2* x2   = reinterpret_cast<const __half2*>(input    + row * hidden_size);
    __half2*       r2   = reinterpret_cast<__half2*>       (residual + row * hidden_size);
    const __half2* w2   = reinterpret_cast<const __half2*>(weight);
    __half2*       y2   = reinterpret_cast<__half2*>       (output   + row * hidden_size);

    int half_size = hidden_size / 2;

    // Pass 1: fused residual add + sum of squares
    float partial = 0.f;
    for (int i = threadIdx.x; i < half_size; i += blockDim.x) {
        __half2 xi = x2[i];
        __half2 ri = r2[i];
        __half2 sum = __hadd2(xi, ri);   // residual add
        r2[i] = sum;                      // write back updated residual

        float a = __half2float(__low2half(sum));
        float b = __half2float(__high2half(sum));
        partial += a * a + b * b;
    }

    partial = warp_reduce_sum(partial);
    if (lane == 0) warp_sums_v4[wid] = partial;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < num_warps) ? warp_sums_v4[lane] : 0.f;
        val = warp_reduce_sum(val);
        if (lane == 0) warp_sums_v4[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(warp_sums_v4[0] / hidden_size + eps);
    __half2 scale2 = __float2half2_rn(rms_inv);

    // Pass 2 (from already-updated residual): normalize + weight
    for (int i = threadIdx.x; i < half_size; i += blockDim.x) {
        __half2 r = r2[i];
        __half2 w = w2[i];
        y2[i] = __hmul2(__hmul2(r, scale2), w);
    }
}

void rmsnorm_v4_fused(
    const __half* input, __half* residual, const __half* weight, __half* output,
    int batch, int hidden_size, float eps, cudaStream_t stream)
{
    int threads = min((hidden_size / 2 + 31) / 32 * 32, 1024);
    int num_warps = threads / 32;
    size_t shmem = num_warps * sizeof(float);
    rmsnorm_kernel_v4_fused<<<batch, threads, shmem, stream>>>(
        input, residual, weight, output, hidden_size, eps);
}
