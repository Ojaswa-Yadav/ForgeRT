// ---------------------------------------------------------------------------
// bench_gemm.cpp – Compare custom vs cuBLAS HGEMM for prefill and decode shapes
//
// Prefill  shape: M=seq_len (large), N=hidden, K=hidden  (e.g. 512×2048×2048)
// Decode   shape: M=1,       N=hidden, K=hidden  (GEMV regime)
//
// Metrics:
//   - Latency (ms)
//   - FLOPs = 2 * M * N * K
//   - Achieved TFLOPS
//   - % of peak FP16 Tensor Core throughput
// ---------------------------------------------------------------------------
#include "types.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <functional>
#include <cstdlib>

struct CudaTimer {
    cudaEvent_t s, e;
    CudaTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~CudaTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void tic() { cudaEventRecord(s); }
    float toc() {
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

static float benchmark(std::function<void()> fn, int warmup=10, int iters=100) {
    for (int i = 0; i < warmup; i++) fn();
    cudaDeviceSynchronize();
    CudaTimer t; t.tic();
    for (int i = 0; i < iters; i++) fn();
    return t.toc() / iters;
}

static fp16* rand_fp16(size_t n) {
    fp16* p; cudaMalloc(&p, n * sizeof(fp16));
    std::vector<fp16> h(n);
    for (size_t i = 0; i < n; i++) h[i] = __float2half((float)rand()/RAND_MAX - 0.5f);
    cudaMemcpy(p, h.data(), n*sizeof(fp16), cudaMemcpyHostToDevice);
    return p;
}

int main() {
    srand(42);
    cublasHandle_t handle; cublasCreate(&handle);

    int dev = 0; cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "GPU: " << prop.name << "\n\n";

    struct Shape { int M, N, K; const char* label; };
    std::vector<Shape> shapes = {
        {   1, 2048, 2048, "decode  Q proj"},
        {   1, 2048, 2048, "decode  K proj"},
        {   8, 2048, 2048, "decode  bs=8  "},
        { 128, 2048, 2048, "prefill T=128 "},
        { 512, 2048, 2048, "prefill T=512 "},
        {1024, 2048, 2048, "prefill T=1024"},
        {   1, 8192, 2048, "decode gate_proj"},
        { 128, 8192, 2048, "prefill gate_proj"},
    };

    std::cout << std::setw(20) << "Shape"
              << std::setw(8)  << "M"
              << std::setw(8)  << "N"
              << std::setw(8)  << "K"
              << std::setw(12) << "cuBLAS(ms)"
              << std::setw(12) << "TFLOPS"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (auto& s : shapes) {
        fp16* A = rand_fp16((size_t)s.M * s.K);
        fp16* B = rand_fp16((size_t)s.N * s.K);
        fp16* C = nullptr; cudaMalloc(&C, (size_t)s.M * s.N * sizeof(fp16));

        __half alpha = __float2half(1.f), beta = __float2half(0.f);

        float ms = benchmark([&]{
            cublasHgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                s.N, s.M, s.K,
                &alpha,
                B, s.K,  // W [N,K] in row-major
                A, s.K,  // x [M,K] in row-major
                &beta,
                C, s.N);
        });

        double flops  = 2.0 * s.M * s.N * s.K;
        double tflops = flops / (ms * 1e-3) / 1e12;

        std::cout << std::setw(20) << s.label
                  << std::setw(8)  << s.M
                  << std::setw(8)  << s.N
                  << std::setw(8)  << s.K
                  << std::setw(12) << std::fixed << std::setprecision(3) << ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << tflops
                  << "\n";

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    cublasDestroy(handle);
    return 0;
}
