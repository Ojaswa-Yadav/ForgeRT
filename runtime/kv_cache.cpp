// ---------------------------------------------------------------------------
// kv_cache.cpp  –  Contiguous KV-cache implementation
// ---------------------------------------------------------------------------
#include "kv_cache.h"
#include <stdexcept>

// ---------------------------------------------------------------------------
// Layout:
//   gpu buffer = [num_layers * 2 * max_seq_len * num_kv_heads * head_dim] fp16
//   layer l:
//     K slice: buf + l * 2 * layer_stride
//     V slice: buf + l * 2 * layer_stride + layer_stride
// ---------------------------------------------------------------------------
KVCache::KVCache(const ModelConfig& config)
{
    cfg = &config;
    // Elements per K (or V) slice for one layer
    layer_stride_ = (size_t)config.max_seq_len
                  * config.num_kv_heads
                  * config.head_dim();

    size_t total_elems = (size_t)config.num_layers * 2 * layer_stride_;
    buf_.alloc(total_elems * sizeof(fp16));

    // Zero-initialize so unused positions don't contain garbage
    CUDA_CHECK(cudaMemset(buf_.ptr, 0, buf_.size));
}

fp16* KVCache::key_cache(int layer_idx) const
{
    return buf_.as<fp16>() + (size_t)layer_idx * 2 * layer_stride_;
}

fp16* KVCache::value_cache(int layer_idx) const
{
    return buf_.as<fp16>() + (size_t)layer_idx * 2 * layer_stride_ + layer_stride_;
}
