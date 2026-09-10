// ---------------------------------------------------------------------------
// model.cpp  –  Weight loading from safetensors format
//
// Safetensors binary layout:
//   [0..7]            : uint64_t header_size  (little-endian)
//   [8 .. 8+header_size) : UTF-8 JSON header
//   [8+header_size ..]  : raw tensor data (densely packed)
//
// JSON schema (per tensor):
//   "tensor_name": {
//       "dtype":         "F16" | "F32" | "BF16" | "I8" | "I4",
//       "shape":         [d0, d1, ...],
//       "data_offsets":  [begin, end]   // bytes within data region
//   }
//
// We cast F32 weights to F16 on the CPU before uploading, and BF16 to F16
// using a simple bit-manipulation (no half-precision BF16 intrinsics needed).
// ---------------------------------------------------------------------------
#include "model.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>   // header-only JSON (included via third_party/)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Safetensors header parser
// ---------------------------------------------------------------------------
std::unordered_map<std::string, TensorMeta>
parse_safetensors_header(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    std::string header_json(header_size, '\0');
    f.read(header_json.data(), header_size);

    auto root = json::parse(header_json);
    std::unordered_map<std::string, TensorMeta> metas;

    for (auto& [name, v] : root.items()) {
        if (name == "__metadata__") continue;
        TensorMeta m;
        m.dtype       = v["dtype"].get<std::string>();
        m.shape       = v["shape"].get<std::vector<int64_t>>();
        m.data_begin  = v["data_offsets"][0].get<int64_t>();
        m.data_end    = v["data_offsets"][1].get<int64_t>();
        metas[name]   = std::move(m);
    }
    return metas;
}

// ---------------------------------------------------------------------------
// BF16 → FP16 conversion (no library dependency)
// BF16 is stored as uint16 with exponent/mantissa matching float but only
// 7 mantissa bits.  Convert: (bf16 << 16) → float, then float → fp16.
// ---------------------------------------------------------------------------
static void bf16_to_fp16(const uint16_t* src, __half* dst, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        uint32_t f32_bits = (uint32_t)src[i] << 16;
        float f;
        std::memcpy(&f, &f32_bits, 4);
        dst[i] = __float2half(f);
    }
}

static void f32_to_fp16(const float* src, __half* dst, size_t n)
{
    for (size_t i = 0; i < n; i++)
        dst[i] = __float2half(src[i]);
}

// ---------------------------------------------------------------------------
// Read raw bytes for one tensor and return an FP16 CPU buffer
// ---------------------------------------------------------------------------
static std::vector<__half> load_tensor_fp16(
    const std::string& path,
    const TensorMeta& meta,
    size_t header_offset)  // = 8 + header_size
{
    int64_t byte_len = meta.data_end - meta.data_begin;
    int64_t n_elems  = 1;
    for (int64_t s : meta.shape) n_elems *= s;

    std::vector<__half> out(n_elems);

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot re-open: " + path);

    int64_t file_offset = header_offset + meta.data_begin;
    f.seekg(file_offset, std::ios::beg);

    if (meta.dtype == "F16") {
        f.read(reinterpret_cast<char*>(out.data()), byte_len);
    } else if (meta.dtype == "BF16") {
        std::vector<uint16_t> buf(n_elems);
        f.read(reinterpret_cast<char*>(buf.data()), byte_len);
        bf16_to_fp16(buf.data(), out.data(), n_elems);
    } else if (meta.dtype == "F32") {
        std::vector<float> buf(n_elems);
        f.read(reinterpret_cast<char*>(buf.data()), byte_len);
        f32_to_fp16(buf.data(), out.data(), n_elems);
    } else {
        throw std::runtime_error("Unsupported dtype for tensor: " + meta.dtype);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Upload FP16 CPU buffer to a pre-allocated device pointer
// ---------------------------------------------------------------------------
static void upload(fp16* dst, const std::vector<fp16>& src)
{
    CUDA_CHECK(cudaMemcpy(dst, src.data(), src.size() * sizeof(fp16), cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// ModelWeights::load
// Expects: {dir}/model.safetensors  (single-file checkpoint)
// For multi-shard checkpoints, provide the first shard; extend as needed.
// ---------------------------------------------------------------------------
std::unique_ptr<ModelWeights> ModelWeights::load(
    const std::string& dir, const ModelConfig& cfg)
{
    // Try common checkpoint filenames
    std::vector<std::string> candidates = {
        dir + "/model.safetensors",
        dir + "/model-00001-of-00001.safetensors",
    };
    std::string path;
    for (auto& c : candidates) {
        std::ifstream test(c);
        if (test) { path = c; break; }
    }
    if (path.empty())
        throw std::runtime_error("No safetensors checkpoint found in: " + dir);

    std::cout << "[ForgeRT] Loading weights from " << path << "\n";

    // Parse header to get sizes
    auto metas = parse_safetensors_header(path);

    // Determine header_offset (needed to seek into data region)
    std::ifstream f(path, std::ios::binary);
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);
    size_t header_offset = 8 + header_size;
    f.close();

    // Helper: load one named tensor as FP16
    auto load = [&](const std::string& key) -> std::vector<fp16> {
        auto it = metas.find(key);
        if (it == metas.end())
            throw std::runtime_error("Tensor not found in checkpoint: " + key);
        return load_tensor_fp16(path, it->second, header_offset);
    };

    // Helper: compute element count from shape
    auto nelems = [&](const std::string& key) -> size_t {
        auto it = metas.find(key);
        if (it == metas.end()) return 0;
        size_t n = 1;
        for (auto s : it->second.shape) n *= s;
        return n;
    };

    // Count total GPU bytes needed
    size_t total_elems = 0;

    // Embeddings + final norm + lm_head
    total_elems += nelems("model.embed_tokens.weight");
    total_elems += nelems("model.norm.weight");
    // lm_head may be tied to embed_tokens
    bool has_lm_head = (metas.count("lm_head.weight") > 0);
    if (has_lm_head) total_elems += nelems("lm_head.weight");

    // Per-layer
    for (int l = 0; l < cfg.num_layers; l++) {
        auto pfx = "model.layers." + std::to_string(l);
        total_elems += nelems(pfx + ".input_layernorm.weight");
        total_elems += nelems(pfx + ".post_attention_layernorm.weight");
        total_elems += nelems(pfx + ".self_attn.q_proj.weight");
        total_elems += nelems(pfx + ".self_attn.k_proj.weight");
        total_elems += nelems(pfx + ".self_attn.v_proj.weight");
        total_elems += nelems(pfx + ".self_attn.o_proj.weight");
        total_elems += nelems(pfx + ".mlp.gate_proj.weight");
        total_elems += nelems(pfx + ".mlp.up_proj.weight");
        total_elems += nelems(pfx + ".mlp.down_proj.weight");
    }

    auto weights = std::unique_ptr<ModelWeights>(new ModelWeights());
    weights->cfg = cfg;
    weights->gpu_pool.alloc(total_elems * sizeof(fp16));
    weights->layers.resize(cfg.num_layers);

    fp16* cursor = weights->gpu_pool.as<fp16>();

    auto upload_tensor = [&](const std::string& key) -> fp16* {
        auto buf = load(key);
        fp16* ptr = cursor;
        cursor += buf.size();
        CUDA_CHECK(cudaMemcpy(ptr, buf.data(), buf.size() * sizeof(fp16), cudaMemcpyHostToDevice));
        return ptr;
    };

    weights->embed_tokens = upload_tensor("model.embed_tokens.weight");
    weights->norm         = upload_tensor("model.norm.weight");
    if (has_lm_head) {
        weights->lm_head = upload_tensor("lm_head.weight");
    } else {
        weights->lm_head = weights->embed_tokens;  // tied
    }

    for (int l = 0; l < cfg.num_layers; l++) {
        auto pfx = "model.layers." + std::to_string(l);
        auto& lw = weights->layers[l];
        lw.input_norm = upload_tensor(pfx + ".input_layernorm.weight");
        lw.post_norm  = upload_tensor(pfx + ".post_attention_layernorm.weight");
        lw.q_proj     = upload_tensor(pfx + ".self_attn.q_proj.weight");
        lw.k_proj     = upload_tensor(pfx + ".self_attn.k_proj.weight");
        lw.v_proj     = upload_tensor(pfx + ".self_attn.v_proj.weight");
        lw.o_proj     = upload_tensor(pfx + ".self_attn.o_proj.weight");
        lw.gate_proj  = upload_tensor(pfx + ".mlp.gate_proj.weight");
        lw.up_proj    = upload_tensor(pfx + ".mlp.up_proj.weight");
        lw.down_proj  = upload_tensor(pfx + ".mlp.down_proj.weight");

        std::cout << "[ForgeRT]   Layer " << l << " loaded\r" << std::flush;
    }
    std::cout << "\n[ForgeRT] All weights loaded.\n";

    return weights;
}
