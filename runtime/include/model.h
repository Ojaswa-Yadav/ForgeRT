#pragma once
// ---------------------------------------------------------------------------
// model.h – Weight containers and safetensors loader
//
// Weights are loaded from a HuggingFace safetensors checkpoint, cast to FP16,
// and kept on the GPU.  CPU memory for the raw file is freed after upload.
// ---------------------------------------------------------------------------
#include "types.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Per-layer weight pointers (views into a single large GPU allocation)
// ---------------------------------------------------------------------------
struct LayerWeights {
    fp16* input_norm   = nullptr;   // [hidden_size]
    fp16* post_norm    = nullptr;   // [hidden_size]
    fp16* q_proj       = nullptr;   // [num_heads * head_dim, hidden_size]
    fp16* k_proj       = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    fp16* v_proj       = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    fp16* o_proj       = nullptr;   // [hidden_size, num_heads * head_dim]
    fp16* gate_proj    = nullptr;   // [intermediate_size, hidden_size]
    fp16* up_proj      = nullptr;   // [intermediate_size, hidden_size]
    fp16* down_proj    = nullptr;   // [hidden_size, intermediate_size]
};

// ---------------------------------------------------------------------------
// Full model weight storage
// ---------------------------------------------------------------------------
class ModelWeights {
public:
    ModelConfig cfg;

    // Single contiguous GPU allocation; layer pointers are views into it.
    DeviceBuffer gpu_pool;

    fp16* embed_tokens  = nullptr;  // [vocab_size, hidden_size]
    fp16* norm          = nullptr;  // [hidden_size]  (final norm)
    fp16* lm_head       = nullptr;  // [vocab_size, hidden_size]

    std::vector<LayerWeights> layers;

    // Load from a directory containing model.safetensors (or shards).
    // Returns a fully-populated ModelWeights on success.
    static std::unique_ptr<ModelWeights> load(const std::string& dir,
                                               const ModelConfig& cfg);

private:
    ModelWeights() = default;
};

// ---------------------------------------------------------------------------
// Safetensors header parser (used internally by model.cpp)
// ---------------------------------------------------------------------------
struct TensorMeta {
    std::string dtype;       // "F16", "F32", "BF16", "I8", ...
    std::vector<int64_t> shape;
    int64_t data_begin = 0;  // byte offset within data region
    int64_t data_end   = 0;
};

std::unordered_map<std::string, TensorMeta>
parse_safetensors_header(const std::string& path);
