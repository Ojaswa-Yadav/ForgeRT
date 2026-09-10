#pragma once
// ---------------------------------------------------------------------------
// executor.h – Transformer forward pass orchestration
// ---------------------------------------------------------------------------
#include "types.h"
#include "model.h"
#include "kv_cache.h"
#include <cublas_v2.h>
#include <memory>

// ---------------------------------------------------------------------------
// Intermediate activation buffers
// ---------------------------------------------------------------------------
struct ActivationBuffers {
    DeviceBuffer hidden;       // [max_seq_len, hidden_size] fp16 – running hidden state
    DeviceBuffer residual;     // spare buffer (used in fused norm experiments)
    DeviceBuffer normed;       // [max_seq_len, hidden_size] fp16
    DeviceBuffer q;            // [max_seq_len, num_heads * head_dim] fp16
    DeviceBuffer k;            // [max_seq_len, num_kv_heads * head_dim] fp16
    DeviceBuffer v;            // [max_seq_len, num_kv_heads * head_dim] fp16
    DeviceBuffer attn_out;     // [max_seq_len, num_heads * head_dim] fp16
    DeviceBuffer gate;         // [max_seq_len, intermediate_size] fp16  (swiglu output)
    DeviceBuffer up;           // [max_seq_len, intermediate_size] fp16
    DeviceBuffer logits;       // [vocab_size] fp32

    void alloc(const ModelConfig& cfg);
};

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------
class Executor {
public:
    Executor(const ModelWeights* weights, KVCache* kv_cache);
    ~Executor();

    // Non-copyable
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    // Prefill: process seq_len tokens from CPU token_ids, populate KV cache.
    // Returns pointer to fp32 logits on the GPU (vocab_size elements).
    fp32* prefill(const int32_t* token_ids_cpu, int seq_len);

    // Decode: process a single new token.
    fp32* decode(int32_t token_id);

    // Greedy argmax over the GPU logit vector.  Synchronizes and returns host int.
    int32_t greedy_sample(const fp32* logits_gpu);

private:
    fp32* run_forward(const int32_t* ids_gpu, int seq_len, int kv_pos, bool is_prefill);
    void  run_layer  (int layer_idx, int seq_len, int kv_pos, bool is_prefill);
    void  run_attention(const LayerWeights& lw, int layer_idx,
                        int seq_len, int kv_pos, bool is_prefill);
    void  run_mlp    (const LayerWeights& lw, int seq_len);

    const ModelWeights* weights_;
    KVCache*            kv_cache_;
    cublasHandle_t      cublas_;
    ActivationBuffers   act_;

    DeviceBuffer argmax_buf_;
    DeviceBuffer argmax_idx_buf_;
};
