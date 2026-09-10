// ---------------------------------------------------------------------------
// bench_attention.cpp – Microbenchmark for all decode attention kernel variants
//
// Sweeps: kv_len, num_heads, head_dim, num_kv_heads (batch=1 always for decode)
//
// Metrics reported:
//   - Latency (CUDA events, warm-up 10 iters, measurement 100 iters)
//   - Achieved memory bandwidth (GB/s)
//     formula: bandwidth = bytes_moved / latency
//     bytes_moved = 2 * kv_len * num_kv_heads * head_dim * 2  (K+V, fp16)
// ---------------------------------------------------------------------------
#include "attention.cuh"
#include "types.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector>
#include <functional>
#include <cstdlib>

// ---------------------------------------------------------------------------
// CUDA event timer
// ---------------------------------------------------------------------------
struct CudaTimer {
    cudaEvent_t s, e;
    CudaTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~CudaTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void tic() { cudaEventRecord(s); }
    float toc() {
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Allocate and fill a random fp16 device buffer
// ---------------------------------------------------------------------------
static fp16* make_rand(size_t n) {
    fp16* ptr;
    cudaMalloc(&ptr, n * sizeof(fp16));
    std::vector<fp16> h(n);
    for (size_t i = 0; i < n; i++) h[i] = __float2half((float)rand() / RAND_MAX - 0.5f);
    cudaMemcpy(ptr, h.data(), n * sizeof(fp16), cudaMemcpyHostToDevice);
    return ptr;
}

// ---------------------------------------------------------------------------
// Benchmark one kernel
// ---------------------------------------------------------------------------
static float benchmark_kernel(
    std::function<void()> fn,
    int warmup = 10, int iters = 100)
{
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();

    CudaTimer t;
    t.tic();
    for (int i = 0; i < iters; i++) fn();
    float ms = t.toc();
    return ms / iters;
}

// ---------------------------------------------------------------------------
// Main sweep
// ---------------------------------------------------------------------------
int main() {
    srand(42);

    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Peak memory bandwidth: "
              << 2.0 * prop.memoryClockRate * 1e3 * prop.memoryBusWidth / 8.0 / 1e9
              << " GB/s\n\n";

    struct Config {
        int num_heads, num_kv_heads, head_dim, kv_len;
    };
    std::vector<Config> configs = {
        {32, 8, 128,  512},
        {32, 8, 128, 1024},
        {32, 8, 128, 2048},
        {32, 8, 128, 4096},
        {32, 8, 128, 8192},
        {16, 4, 128, 2048},
        {32, 32, 128, 2048},  // MHA (no GQA)
    };

    std::cout << std::setw(6)  << "nH"
              << std::setw(6)  << "nKV"
              << std::setw(6)  << "hd"
              << std::setw(8)  << "kv_len"
              << std::setw(10) << "v0(ms)"
              << std::setw(10) << "v1(ms)"
              << std::setw(10) << "v2(ms)"
              << std::setw(10) << "v3(ms)"
              << std::setw(10) << "v4(ms)"
              << std::setw(10) << "BW v4(GB/s)"
              << "\n";
    std::cout << std::string(86, '-') << "\n";

    for (auto& c : configs) {
        fp16* q   = make_rand(c.num_heads    * c.head_dim);  // decode: 1 token
        fp16* kc  = make_rand((size_t)c.kv_len * c.num_kv_heads * c.head_dim);
        fp16* vc  = make_rand((size_t)c.kv_len * c.num_kv_heads * c.head_dim);
        fp16* out = nullptr;
        cudaMalloc(&out, (size_t)c.num_heads * c.head_dim * sizeof(fp16));

        // v0 – skip for large kv_len (uses dynamic shared mem proportional to kv_len)
        float t0 = -1.f;
        if (c.kv_len <= 2048) {
            t0 = benchmark_kernel([&]{
                decode_attn_v0_naive(q, kc, vc, out, c.num_heads, c.num_kv_heads, c.head_dim, c.kv_len);
            }, 5, 20);
        }

        float t1 = benchmark_kernel([&]{
            decode_attn_v1_coal(q, kc, vc, out, c.num_heads, c.num_kv_heads, c.head_dim, c.kv_len);
        });
        float t2 = benchmark_kernel([&]{
            decode_attn_v2_warp_red(q, kc, vc, out, c.num_heads, c.num_kv_heads, c.head_dim, c.kv_len);
        });
        float t3 = benchmark_kernel([&]{
            decode_attn_v3_vec(q, kc, vc, out, c.num_heads, c.num_kv_heads, c.head_dim, c.kv_len);
        });
        float t4 = benchmark_kernel([&]{
            decode_attn_v4_flash(q, kc, vc, out, c.num_heads, c.num_kv_heads, c.head_dim, c.kv_len);
        });

        // Bytes moved: load K + V caches
        double bytes = 2.0 * c.kv_len * c.num_kv_heads * c.head_dim * sizeof(fp16);
        // Q is tiny (1 token), output is tiny: dominated by K+V loads.
        double bw_v4 = bytes / (t4 * 1e-3) / 1e9;

        std::cout << std::setw(6)  << c.num_heads
                  << std::setw(6)  << c.num_kv_heads
                  << std::setw(6)  << c.head_dim
                  << std::setw(8)  << c.kv_len;
        if (t0 < 0) std::cout << std::setw(10) << "skip";
        else         std::cout << std::setw(10) << std::fixed << std::setprecision(3) << t0;
        std::cout << std::setw(10) << std::fixed << std::setprecision(3) << t1
                  << std::setw(10) << std::fixed << std::setprecision(3) << t2
                  << std::setw(10) << std::fixed << std::setprecision(3) << t3
                  << std::setw(10) << std::fixed << std::setprecision(3) << t4
                  << std::setw(10) << std::fixed << std::setprecision(1) << bw_v4
                  << "\n";

        cudaFree(q); cudaFree(kc); cudaFree(vc); cudaFree(out);
    }

    return 0;
}
