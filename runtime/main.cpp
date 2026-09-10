// ---------------------------------------------------------------------------
// main.cpp – ForgeRT CLI
//
// Usage:
//   forgert <model_dir> [--prompt "..."] [--max_tokens N] [--temperature T]
//
// model_dir must contain:
//   config.json            (HuggingFace model config)
//   model.safetensors      (or model-00001-of-00001.safetensors)
//   tokenizer.json         (HuggingFace fast tokenizer)
//
// Phase 1: greedy decode only.
// ---------------------------------------------------------------------------
#include "model.h"
#include "executor.h"
#include "kv_cache.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Minimal config.json reader (HuggingFace format)
// ---------------------------------------------------------------------------
static ModelConfig load_model_config(const std::string& dir)
{
    std::ifstream f(dir + "/config.json");
    if (!f) throw std::runtime_error("Cannot open config.json in: " + dir);

    json cfg = json::parse(f);
    ModelConfig c;

    c.vocab_size       = cfg.value("vocab_size",        32000);
    c.hidden_size      = cfg.value("hidden_size",        2048);
    c.num_heads        = cfg.value("num_attention_heads", 32);
    c.num_kv_heads     = cfg.value("num_key_value_heads", 8);
    c.num_layers       = cfg.value("num_hidden_layers",  22);
    c.intermediate_size= cfg.value("intermediate_size", 8192);
    c.max_seq_len      = cfg.value("max_position_embeddings", 4096);
    c.rms_norm_eps     = cfg.value("rms_norm_eps",      1e-5f);
    c.rope_base        = cfg.value("rope_theta",        500000.f);
    c.tie_embeddings   = cfg.value("tie_word_embeddings", false);

    std::cout << "[ForgeRT] Model config:\n"
              << "  hidden_size      = " << c.hidden_size      << "\n"
              << "  num_layers       = " << c.num_layers       << "\n"
              << "  num_heads        = " << c.num_heads        << "\n"
              << "  num_kv_heads     = " << c.num_kv_heads     << "\n"
              << "  intermediate     = " << c.intermediate_size<< "\n"
              << "  vocab_size       = " << c.vocab_size       << "\n"
              << "  head_dim         = " << c.head_dim()       << "\n"
              << "  rope_base        = " << c.rope_base        << "\n";
    return c;
}

// ---------------------------------------------------------------------------
// Minimal BPE tokenizer wrapper using tokenizer.json
// For a proper implementation, use the HuggingFace tokenizers C++ library.
// Here we implement a simple encode/decode via the Python subprocess fallback
// (see python/tokenize_helper.py).
// This stub lets the runtime compile; replace with the real tokenizer.
// ---------------------------------------------------------------------------
struct SimpleTokenizer {
    json tok_json;
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string> id_to_token;
    int32_t bos_id = 1, eos_id = 2;

    explicit SimpleTokenizer(const std::string& dir) {
        std::ifstream f(dir + "/tokenizer.json");
        if (!f) {
            std::cerr << "[Warning] tokenizer.json not found; using dummy tokenizer\n";
            return;
        }
        tok_json = json::parse(f);

        // Build vocab from the "model.vocab" map (tiktoken / sentencepiece style)
        if (tok_json.contains("model") && tok_json["model"].contains("vocab")) {
            for (auto& [k, v] : tok_json["model"]["vocab"].items()) {
                int id = v.get<int>();
                vocab[k] = id;
                if ((int)id_to_token.size() <= id)
                    id_to_token.resize(id + 1);
                id_to_token[id] = k;
            }
        }
        if (tok_json.contains("model") && tok_json["model"].contains("type")) {
            std::cout << "[ForgeRT] Tokenizer type: " << tok_json["model"]["type"] << "\n";
        }
    }

    // Naive word-level encode (placeholder; real BPE needed for accurate output)
    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> ids = { bos_id };
        // Split on space and look up each word (not a real BPE — use Python bridge)
        // For CLI use, call `python/tokenize_helper.py` or link tokenizers-cpp.
        std::cerr << "[Warning] Using placeholder tokenizer — output will be meaningless.\n"
                  << "  Run:  python python/tokenize.py --model <dir> --text '" << text << "'\n"
                  << "  to get real token ids.\n";
        ids.push_back(eos_id);
        return ids;
    }

    std::string decode(const std::vector<int32_t>& ids) const {
        std::string out;
        for (int32_t id : ids) {
            if (id < (int)id_to_token.size() && !id_to_token[id].empty()) {
                // BPE tokens store ▁ (U+2581) for leading space; replace
                std::string tok = id_to_token[id];
                for (size_t i = 0; i < tok.size(); ) {
                    unsigned char c = tok[i];
                    if (c == 0xe2 && i+2 < tok.size() &&
                        (unsigned char)tok[i+1] == 0x96 &&
                        (unsigned char)tok[i+2] == 0x81) {
                        out += ' ';
                        i += 3;
                    } else {
                        out += tok[i++];
                    }
                }
            }
        }
        return out;
    }
};

// ---------------------------------------------------------------------------
// Latency measurement helper
// ---------------------------------------------------------------------------
struct Timer {
    cudaEvent_t start_, stop_;
    Timer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~Timer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void tic() { cudaEventRecord(start_); }
    float toc() {  // returns ms
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string model_dir  = ".";
    std::string prompt     = "The meaning of life is";
    int         max_tokens = 50;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--prompt")     == 0 && i+1 < argc) prompt     = argv[++i];
        else if (std::strcmp(argv[i], "--max_tokens") == 0 && i+1 < argc) max_tokens = std::stoi(argv[++i]);
        else if (argv[i][0] != '-') model_dir = argv[i];
    }

    // Print GPU info
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "[ForgeRT] GPU: " << prop.name
              << "  VRAM: " << prop.totalGlobalMem / (1<<20) << " MiB\n";

    // Load config + weights
    ModelConfig cfg = load_model_config(model_dir);

    auto t0 = std::chrono::steady_clock::now();
    auto weights = ModelWeights::load(model_dir, cfg);
    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout << "[ForgeRT] Weight load: " << load_ms << " ms\n";

    // KV cache + executor
    KVCache   kv_cache(cfg);
    Executor  executor(weights.get(), &kv_cache);

    // Tokenize
    SimpleTokenizer tok(model_dir);
    std::vector<int32_t> input_ids = tok.encode(prompt);
    std::cout << "[ForgeRT] Prompt tokens: " << input_ids.size() << "\n";

    // --- Prefill ---
    Timer timer;
    timer.tic();
    fp32* logits = executor.prefill(input_ids.data(), (int)input_ids.size());
    float prefill_ms = timer.toc();
    std::cout << "[ForgeRT] Prefill latency: " << prefill_ms << " ms  ("
              << input_ids.size() / (prefill_ms / 1000.f) << " tokens/s)\n";

    int32_t next_tok = executor.greedy_sample(logits);
    std::cout << "\n--- Generated ---\n";
    std::cout << prompt;

    std::vector<int32_t> generated;
    generated.push_back(next_tok);

    // --- Decode loop ---
    float total_decode_ms = 0.f;
    for (int step = 0; step < max_tokens - 1; step++) {
        timer.tic();
        logits = executor.decode(next_tok);
        float step_ms = timer.toc();
        total_decode_ms += step_ms;

        next_tok = executor.greedy_sample(logits);
        generated.push_back(next_tok);

        // Print token as it's generated
        std::string tok_str = tok.decode({next_tok});
        std::cout << tok_str << std::flush;

        if (next_tok == tok.eos_id) break;
    }

    std::cout << "\n\n";
    float avg_decode_ms = total_decode_ms / generated.size();
    std::cout << "[ForgeRT] Decode: " << generated.size() << " tokens"
              << "  avg " << avg_decode_ms << " ms/tok"
              << "  (" << 1000.f / avg_decode_ms << " tokens/s)\n";

    // VRAM usage
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "[ForgeRT] VRAM used: "
              << (total_mem - free_mem) / (1<<20) << " MiB / "
              << total_mem / (1<<20) << " MiB\n";

    return 0;
}
