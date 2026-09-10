// ---------------------------------------------------------------------------
// executor.cpp  –  Transformer forward pass
//
// Standard pre-norm transformer block:
//   normed   = rms_norm(hidden)
//   attn_out = attention(normed)  then o_proj
//   hidden   = hidden + attn_out                // residual
//   normed   = rms_norm(hidden)
//   mlp_out  = mlp(normed)
//   hidden   = hidden + mlp_out                 // residual
//
// hidden is the sole running hidden-state tensor.
// The fused rmsnorm_v4_fused (residual add + norm in one pass) is used in
// Phase 2; Phase 1 uses explicit residual adds for clarity and correctness.
// ---------------------------------------------------------------------------
#include "executor.h"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "attention.cuh"
#include "elementwise.cuh"
#include "embed.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

// greedy_sample is defined in sampling.cu
struct MaxPair { float val; int32_t idx; };
extern void greedy_sample(const float*, MaxPair*, int32_t*, int, cudaStream_t);

// Precomputed RoPE tables (built once, shared across all layers)
static float* g_cos_table = nullptr;
static float* g_sin_table = nullptr;

// ---------------------------------------------------------------------------
// ActivationBuffers
// ---------------------------------------------------------------------------
void ActivationBuffers::alloc(const ModelConfig& cfg)
{
    size_t T  = cfg.max_seq_len;
    size_t H  = cfg.hidden_size;
    size_t I  = cfg.intermediate_size;
    size_t nH = cfg.num_heads;
    size_t nK = cfg.num_kv_heads;
    size_t hd = cfg.head_dim();

    hidden  .alloc(T * H * sizeof(fp16));
    residual.alloc(T * H * sizeof(fp16));    // re-used as scratch for normed output
    normed  .alloc(T * H * sizeof(fp16));
    q       .alloc(T * nH * hd * sizeof(fp16));
    k       .alloc(T * nK * hd * sizeof(fp16));
    v       .alloc(T * nK * hd * sizeof(fp16));
    attn_out.alloc(T * H * sizeof(fp16));
    gate    .alloc(T * I * sizeof(fp16));
    up      .alloc(T * I * sizeof(fp16));
    logits  .alloc(cfg.vocab_size * sizeof(fp32));
}

// ---------------------------------------------------------------------------
// hgemm: y[M, N] = x[M, K] * W[N, K]^T   (all row-major, cuBLAS col-major trick)
//
// cuBLAS is column-major.  Row-major [M, K] looks like col-major [K, M].
// We compute:  y^T[N,M] = W[N,K] * x^T[K,M]
// i.e.  cublasHgemm with opA=OP_T on W, opB=OP_N on x.
// ---------------------------------------------------------------------------
static void hgemm(cublasHandle_t handle,
                  const fp16* x, const fp16* W, fp16* y,
                  int M, int N, int K,
                  float alpha = 1.f, float beta = 0.f)
{
    __half alpha_h = __float2half(alpha);
    __half beta_h  = __float2half(beta);

    // op(A) = W (row-major [N,K] seen as col-major [K,N], OP_T gives [N,K])
    // op(B) = x^T (x row-major [M,K] seen as col-major [K,M], OP_N gives [K,M])
    // C     = y^T (y row-major [M,N] seen as col-major [N,M])
    CUBLAS_CHECK(cublasHgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha_h,
        W, K,   // A = W col-major [K, N], lda = K
        x, K,   // B = x col-major [K, M], ldb = K
        &beta_h,
        y, N)); // C = y col-major [N, M], ldc = N
}

// ---------------------------------------------------------------------------
// Executor constructor / destructor
// ---------------------------------------------------------------------------
Executor::Executor(const ModelWeights* weights, KVCache* kv_cache)
    : weights_(weights), kv_cache_(kv_cache)
{
    CUBLAS_CHECK(cublasCreate(&cublas_));
    act_.alloc(weights->cfg);

    if (!g_cos_table) {
        rope_precompute_tables(
            &g_cos_table, &g_sin_table,
            weights->cfg.max_seq_len,
            weights->cfg.head_dim(),
            weights->cfg.rope_base,
            weights->cfg.rope_scale);
    }

    // Allocate argmax scratch: [1024 blocks] MaxPair + 1 int32
    argmax_buf_    .alloc(1024 * (sizeof(float) + sizeof(int32_t)));
    argmax_idx_buf_.alloc(sizeof(int32_t));
}

Executor::~Executor()
{
    cublasDestroy(cublas_);
}

// ---------------------------------------------------------------------------
// run_attention
// ---------------------------------------------------------------------------
void Executor::run_attention(const LayerWeights& lw, int layer_idx,
                              int seq_len, int kv_pos, bool is_prefill)
{
    const ModelConfig& cfg = weights_->cfg;
    int H   = cfg.hidden_size;
    int hd  = cfg.head_dim();
    int nH  = cfg.num_heads;
    int nK  = cfg.num_kv_heads;

    // Q, K, V projections on normed hidden
    hgemm(cublas_, act_.normed.as<fp16>(), lw.q_proj, act_.q.as<fp16>(), seq_len, nH * hd, H);
    hgemm(cublas_, act_.normed.as<fp16>(), lw.k_proj, act_.k.as<fp16>(), seq_len, nK * hd, H);
    hgemm(cublas_, act_.normed.as<fp16>(), lw.v_proj, act_.v.as<fp16>(), seq_len, nK * hd, H);

    // RoPE in-place on Q and K
    rope_v2_fused(act_.q.as<fp16>(), act_.k.as<fp16>(),
                  g_cos_table, g_sin_table,
                  seq_len, nH, nK, hd, kv_pos);

    // Append new K, V into KV cache at position kv_pos
    CUDA_CHECK(cudaMemcpyAsync(
        kv_cache_->key_cache(layer_idx)   + (size_t)kv_pos * nK * hd,
        act_.k.as<fp16>(),
        (size_t)seq_len * nK * hd * sizeof(fp16), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpyAsync(
        kv_cache_->value_cache(layer_idx) + (size_t)kv_pos * nK * hd,
        act_.v.as<fp16>(),
        (size_t)seq_len * nK * hd * sizeof(fp16), cudaMemcpyDeviceToDevice));

    int kv_len = kv_pos + seq_len;

    // Attention
    if (is_prefill) {
        prefill_attn_causal(
            act_.q.as<fp16>(),
            kv_cache_->key_cache(layer_idx),
            kv_cache_->value_cache(layer_idx),
            act_.attn_out.as<fp16>(),
            seq_len, nH, nK, hd);
    } else {
        decode_attn_v4_flash(
            act_.q.as<fp16>(),
            kv_cache_->key_cache(layer_idx),
            kv_cache_->value_cache(layer_idx),
            act_.attn_out.as<fp16>(),
            nH, nK, hd, kv_len);
    }

    // O projection: attn_out is [seq_len, nH*hd] → [seq_len, H]
    // We accumulate into hidden with beta=1 (add to existing hidden value).
    // hidden currently holds the pre-attention hidden state (the skip connection).
    hgemm(cublas_, act_.attn_out.as<fp16>(), lw.o_proj, act_.hidden.as<fp16>(),
          seq_len, H, nH * hd, 1.f, /*beta=*/1.f);
    // Now act_.hidden = hidden_pre + o_proj(attn_out)  [residual already added]
}

// ---------------------------------------------------------------------------
// run_mlp
// ---------------------------------------------------------------------------
void Executor::run_mlp(const LayerWeights& lw, int seq_len)
{
    const ModelConfig& cfg = weights_->cfg;
    int H  = cfg.hidden_size;
    int I  = cfg.intermediate_size;

    hgemm(cublas_, act_.normed.as<fp16>(), lw.gate_proj, act_.gate.as<fp16>(), seq_len, I, H);
    hgemm(cublas_, act_.normed.as<fp16>(), lw.up_proj,   act_.up.as<fp16>(),   seq_len, I, H);
    swiglu_inplace(act_.gate.as<fp16>(), act_.up.as<fp16>(), seq_len, I);
    // down_proj accumulates into hidden (beta=1)
    hgemm(cublas_, act_.gate.as<fp16>(), lw.down_proj, act_.hidden.as<fp16>(),
          seq_len, H, I, 1.f, /*beta=*/1.f);
    // Now act_.hidden = hidden_post_attn + down_proj(swiglu)  [MLP residual added]
}

// ---------------------------------------------------------------------------
// run_layer
// Hidden state throughout: act_.hidden  (single buffer, residuals added in-place)
// ---------------------------------------------------------------------------
void Executor::run_layer(int layer_idx, int seq_len, int kv_pos, bool is_prefill)
{
    const ModelConfig& cfg = weights_->cfg;
    const LayerWeights& lw = weights_->layers[layer_idx];
    int H = cfg.hidden_size;

    // 1. Attention pre-norm: normed = rms_norm(hidden)
    rmsnorm_v2_warp(act_.hidden.as<fp16>(), lw.input_norm, act_.normed.as<fp16>(),
                    seq_len, H, cfg.rms_norm_eps);

    // 2. Attention + O proj + residual add (beta=1 in hgemm absorbs residual)
    //    Before call: hidden = skip connection value
    //    After call:  hidden = hidden + attn_out
    run_attention(lw, layer_idx, seq_len, kv_pos, is_prefill);

    // 3. MLP pre-norm: normed = rms_norm(hidden)
    rmsnorm_v2_warp(act_.hidden.as<fp16>(), lw.post_norm, act_.normed.as<fp16>(),
                    seq_len, H, cfg.rms_norm_eps);

    // 4. MLP + residual add (beta=1 in hgemm absorbs residual)
    run_mlp(lw, seq_len);
    // After: hidden = hidden_post_attn + mlp_out
}

// ---------------------------------------------------------------------------
// run_forward  –  shared body of prefill and decode
// token_ids_gpu: device int32 [seq_len]
// ---------------------------------------------------------------------------
fp32* Executor::run_forward(const int32_t* ids_gpu, int seq_len, int kv_pos, bool is_prefill)
{
    const ModelConfig& cfg = weights_->cfg;
    int H = cfg.hidden_size;

    // Embed tokens via GPU gather kernel (embed.cu)
    embed_gather(weights_->embed_tokens, ids_gpu, act_.hidden.as<fp16>(), seq_len, H);

    // Transformer layers
    for (int l = 0; l < cfg.num_layers; l++) {
        run_layer(l, seq_len, kv_pos, is_prefill);
    }

    // Final RMSNorm on the last token
    int last_tok_offset = (seq_len - 1) * H;
    rmsnorm_v2_warp(
        act_.hidden.as<fp16>() + last_tok_offset,
        weights_->norm,
        act_.normed.as<fp16>(),
        1, H, cfg.rms_norm_eps);

    // LM head: logits [1, vocab_size] = normed [1, H] * lm_head[vocab_size, H]^T
    // Use fp32 accumulation.
    {
        const fp32 alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            cfg.vocab_size, 1, H,
            &alpha,
            weights_->lm_head, CUDA_R_16F, H,   // lm_head [vocab, H] as col-major [H, vocab]
            act_.normed.as<fp16>(), CUDA_R_16F, H,
            &beta,
            act_.logits.as<fp32>(), CUDA_R_32F, cfg.vocab_size,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    return act_.logits.as<fp32>();
}

// ---------------------------------------------------------------------------
// prefill
// ---------------------------------------------------------------------------
fp32* Executor::prefill(const int32_t* token_ids_cpu, int seq_len)
{
    DeviceBuffer ids_gpu(seq_len * sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(ids_gpu.ptr, token_ids_cpu,
                          seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));
    kv_cache_->seq_len = 0;
    fp32* logits = run_forward(ids_gpu.as<int32_t>(), seq_len, 0, true);
    kv_cache_->seq_len = seq_len;
    return logits;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
fp32* Executor::decode(int32_t token_id)
{
    int kv_pos = kv_cache_->seq_len;
    DeviceBuffer id_gpu(sizeof(int32_t));
    CUDA_CHECK(cudaMemcpy(id_gpu.ptr, &token_id, sizeof(int32_t), cudaMemcpyHostToDevice));
    fp32* logits = run_forward(id_gpu.as<int32_t>(), 1, kv_pos, false);
    kv_cache_->seq_len++;
    return logits;
}

// ---------------------------------------------------------------------------
// greedy_sample
// ---------------------------------------------------------------------------
int32_t Executor::greedy_sample(const fp32* logits_gpu)
{
    ::greedy_sample(logits_gpu,
                    reinterpret_cast<MaxPair*>(argmax_buf_.ptr),
                    argmax_idx_buf_.as<int32_t>(),
                    weights_->cfg.vocab_size,
                    nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    int32_t tok;
    CUDA_CHECK(cudaMemcpy(&tok, argmax_idx_buf_.ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
    return tok;
}
