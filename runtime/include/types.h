#pragma once
// ---------------------------------------------------------------------------
// types.h - Common type aliases and GPU tensor handle
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

// ---- type aliases ----------------------------------------------------------
using fp16 = __half;
using fp32 = float;

// ---- CUDA error check -------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at ")               \
                + __FILE__ + ":" + std::to_string(__LINE__) + " — "             \
                + cudaGetErrorString(err_));                                      \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t st_ = (call);                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(std::string("cuBLAS error at ")             \
                + __FILE__ + ":" + std::to_string(__LINE__)                      \
                + " code=" + std::to_string(static_cast<int>(st_)));             \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// DeviceBuffer – RAII wrapper around a raw GPU allocation.
// ---------------------------------------------------------------------------
struct DeviceBuffer {
    void*  ptr  = nullptr;
    size_t size = 0;   // bytes

    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t bytes) { alloc(bytes); }

    ~DeviceBuffer() { free(); }

    // Non-copyable, movable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr; o.size = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { free(); ptr = o.ptr; size = o.size; o.ptr = nullptr; o.size = 0; }
        return *this;
    }

    void alloc(size_t bytes) {
        free();
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        size = bytes;
    }

    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; size = 0; }
    }

    template<typename T> T*       as()       { return static_cast<T*>(ptr); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr); }
};

// ---------------------------------------------------------------------------
// ModelConfig – architecture hyperparameters (Llama/Qwen-style)
// ---------------------------------------------------------------------------
struct ModelConfig {
    int32_t vocab_size      = 32000;
    int32_t hidden_size     = 2048;
    int32_t num_heads       = 32;
    int32_t num_kv_heads    = 8;    // GQA: typically < num_heads
    int32_t num_layers      = 22;
    int32_t intermediate_size = 8192;
    int32_t max_seq_len     = 4096;
    float   rms_norm_eps    = 1e-5f;
    float   rope_base       = 500000.0f;  // Llama 3 uses 500k
    float   rope_scale      = 1.0f;       // NTK / YaRN scale factor
    bool    tie_embeddings  = false;

    int32_t head_dim() const { return hidden_size / num_heads; }
    int32_t gqa_ratio() const { return num_heads / num_kv_heads; }
};
