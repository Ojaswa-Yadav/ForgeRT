#pragma once
// ---------------------------------------------------------------------------
// kv_cache.h – Contiguous KV-cache for single-request autoregressive decode
//
// Layout (per layer, contiguous block):
//   K cache: [max_seq_len, num_kv_heads, head_dim]  fp16
//   V cache: [max_seq_len, num_kv_heads, head_dim]  fp16
//
// All layers share one large GPU allocation; each layer gets a fixed-stride
// window into it.
// ---------------------------------------------------------------------------
#include "types.h"
#include <vector>

class KVCache {
public:
    KVCache() = default;
    KVCache(const ModelConfig& cfg);

    // Pointers into the flat GPU buffer for layer `layer_idx`.
    fp16* key_cache  (int layer_idx) const;
    fp16* value_cache(int layer_idx) const;

    // Current number of valid tokens stored (updated externally by executor).
    int seq_len = 0;

    const ModelConfig* cfg = nullptr;

private:
    DeviceBuffer buf_;       // single allocation for all layers
    size_t layer_stride_;    // elements per K-or-V slice
};
