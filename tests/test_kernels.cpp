// ---------------------------------------------------------------------------
// test_kernels.cpp
//
// Correctness tests comparing ForgeRT CUDA kernels against expected outputs.
// Each test allocates random fp16 data, runs the kernel, downloads results,
// and checks against a CPU reference implementation within tolerance.
//
// Run after build:  ./test_kernels
// Expected output:  all PASS lines.
// ---------------------------------------------------------------------------
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "attention.cuh"
#include "elementwise.cuh"
#include "types.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <string>

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int g_tests = 0, g_fails = 0;

static void check(const std::string& name, bool passed) {
    g_tests++;
    if (!passed) { g_fails++; std::cout << "[FAIL] " << name << "\n"; }
    else          std::cout << "[PASS] " << name << "\n";
}

static bool allclose(const std::vector<float>& a, const std::vector<float>& b, float atol=1e-2f) {
    if (a.size() != b.size()) return false;
    float max_err = 0.f;
    for (size_t i = 0; i < a.size(); i++) {
        float err = std::abs(a[i] - b[i]);
        max_err = std::max(max_err, err);
        if (err > atol) {
            std::cerr << "  max_err=" << max_err << " at i=" << i
                      << " got=" << a[i] << " ref=" << b[i] << "\n";
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------
static std::vector<float> cpu_rmsnorm(
    const std::vector<float>& x, const std::vector<float>& w,
    int batch, int H, float eps)
{
    std::vector<float> out(batch * H);
    for (int b = 0; b < batch; b++) {
        float ss = 0.f;
        for (int i = 0; i < H; i++) ss += x[b*H+i] * x[b*H+i];
        float rms_inv = 1.f / std::sqrt(ss / H + eps);
        for (int i = 0; i < H; i++)
            out[b*H+i] = x[b*H+i] * rms_inv * w[i];
    }
    return out;
}

static std::vector<float> cpu_swiglu(
    std::vector<float> gate, const std::vector<float>& up, int n)
{
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu_g = g / (1.f + std::exp(-g));
        gate[i] = silu_g * up[i];
    }
    return gate;
}

// ---------------------------------------------------------------------------
// Helpers: fp32 ↔ GPU fp16
// ---------------------------------------------------------------------------
static fp16* upload_fp16(const std::vector<float>& h) {
    std::vector<fp16> hh(h.size());
    for (size_t i = 0; i < h.size(); i++) hh[i] = __float2half(h[i]);
    fp16* d; cudaMalloc(&d, hh.size()*sizeof(fp16));
    cudaMemcpy(d, hh.data(), hh.size()*sizeof(fp16), cudaMemcpyHostToDevice);
    return d;
}

static std::vector<float> download_fp16(const fp16* d, size_t n) {
    std::vector<fp16> hh(n); cudaMemcpy(hh.data(), d, n*sizeof(fp16), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; i++) out[i] = __half2float(hh[i]);
    return out;
}

static std::vector<float> rand_vec(size_t n, float lo=-1.f, float hi=1.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = lo + (hi-lo) * (float)rand()/RAND_MAX;
    return v;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
static void test_rmsnorm() {
    const int batch = 4, H = 256;
    auto x = rand_vec(batch * H);
    auto w = rand_vec(H, 0.8f, 1.2f);  // weights near 1

    auto ref = cpu_rmsnorm(x, w, batch, H, 1e-5f);

    fp16* dx = upload_fp16(x);
    fp16* dw = upload_fp16(w);
    fp16* dy; cudaMalloc(&dy, batch * H * sizeof(fp16));

    // v1_block
    rmsnorm_v1_block(dx, dw, dy, batch, H, 1e-5f);
    cudaDeviceSynchronize();
    auto out1 = download_fp16(dy, batch * H);
    check("rmsnorm_v1_block", allclose(out1, ref));

    // v2_warp
    rmsnorm_v2_warp(dx, dw, dy, batch, H, 1e-5f);
    cudaDeviceSynchronize();
    auto out2 = download_fp16(dy, batch * H);
    check("rmsnorm_v2_warp", allclose(out2, ref));

    // v3_vec
    rmsnorm_v3_vec(dx, dw, dy, batch, H, 1e-5f);
    cudaDeviceSynchronize();
    auto out3 = download_fp16(dy, batch * H);
    check("rmsnorm_v3_vec", allclose(out3, ref));

    cudaFree(dx); cudaFree(dw); cudaFree(dy);
}

static void test_rmsnorm_fused() {
    const int batch = 2, H = 128;
    auto x   = rand_vec(batch * H);
    auto r   = rand_vec(batch * H);
    auto w   = rand_vec(H, 0.9f, 1.1f);

    // CPU ref: fused = rms_norm(x + r)
    std::vector<float> xr(batch * H);
    for (int i = 0; i < batch*H; i++) xr[i] = x[i] + r[i];
    auto ref = cpu_rmsnorm(xr, w, batch, H, 1e-5f);

    fp16* dx = upload_fp16(x);
    fp16* dr = upload_fp16(r);
    fp16* dw = upload_fp16(w);
    fp16* dy; cudaMalloc(&dy, batch * H * sizeof(fp16));

    rmsnorm_v4_fused(dx, dr, dw, dy, batch, H, 1e-5f);
    cudaDeviceSynchronize();
    auto out = download_fp16(dy, batch * H);
    check("rmsnorm_v4_fused (output)", allclose(out, ref));

    // Also check that residual was updated in-place
    auto updated_r = download_fp16(dr, batch * H);
    check("rmsnorm_v4_fused (residual updated)", allclose(updated_r, xr));

    cudaFree(dx); cudaFree(dr); cudaFree(dw); cudaFree(dy);
}

static void test_swiglu() {
    const int batch = 3, I = 256;
    int n = batch * I;
    auto gate = rand_vec(n);
    auto up   = rand_vec(n);
    auto ref  = cpu_swiglu(gate, up, n);

    fp16* dg = upload_fp16(gate);
    fp16* du = upload_fp16(up);

    swiglu_inplace(dg, du, batch, I);
    cudaDeviceSynchronize();
    auto out = download_fp16(dg, n);
    check("swiglu_inplace", allclose(out, ref, 2e-2f));

    cudaFree(dg); cudaFree(du);
}

static void test_residual_add() {
    const int n = 512;
    auto a = rand_vec(n), b = rand_vec(n);
    std::vector<float> ref(n);
    for (int i = 0; i < n; i++) ref[i] = a[i] + b[i];

    fp16* da = upload_fp16(a);
    fp16* db = upload_fp16(b);
    residual_add(da, db, n);
    cudaDeviceSynchronize();
    auto out = download_fp16(da, n);
    check("residual_add", allclose(out, ref));

    cudaFree(da); cudaFree(db);
}

static void test_rope_tables() {
    float* cos_tab; float* sin_tab;
    rope_precompute_tables(&cos_tab, &sin_tab, 64, 64, 10000.f, 1.f);
    cudaDeviceSynchronize();

    // Download and spot-check: cos(0) = 1, sin(0) = 0 for all frequencies
    std::vector<float> cos_h(64*32), sin_h(64*32);
    cudaMemcpy(cos_h.data(), cos_tab, cos_h.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sin_h.data(), sin_tab, sin_h.size()*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = true;
    for (int f = 0; f < 32; f++) {
        // Position 0: all cos=1, sin=0
        if (std::abs(cos_h[f] - 1.f) > 1e-5f) { ok = false; break; }
        if (std::abs(sin_h[f] - 0.f) > 1e-5f) { ok = false; break; }
    }
    check("rope_tables (pos=0 values)", ok);

    cudaFree(cos_tab); cudaFree(sin_tab);
}

// ---------------------------------------------------------------------------
// CPU reference: decode attention with GQA
// q:       [num_heads, head_dim]
// k_cache: [kv_len, num_kv_heads, head_dim]
// v_cache: [kv_len, num_kv_heads, head_dim]
// ---------------------------------------------------------------------------
static std::vector<float> cpu_decode_attn(
    const std::vector<float>& q,       // [num_heads * head_dim]
    const std::vector<float>& k_cache, // [kv_len * num_kv_heads * head_dim]
    const std::vector<float>& v_cache,
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    int gqa = num_heads / num_kv_heads;
    float scale = 1.f / std::sqrt((float)head_dim);
    std::vector<float> out(num_heads * head_dim, 0.f);

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / gqa;
        const float* q_h = q.data() + h * head_dim;

        // Compute all scores
        std::vector<float> scores(kv_len);
        float max_s = -1e30f;
        for (int t = 0; t < kv_len; t++) {
            const float* k_t = k_cache.data() + (t * num_kv_heads + kv_h) * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * scale;
            max_s = std::max(max_s, scores[t]);
        }

        // Softmax
        float denom = 0.f;
        for (int t = 0; t < kv_len; t++) {
            scores[t] = std::exp(scores[t] - max_s);
            denom += scores[t];
        }

        // Weighted sum over V
        float* o = out.data() + h * head_dim;
        for (int t = 0; t < kv_len; t++) {
            const float* v_t = v_cache.data() + (t * num_kv_heads + kv_h) * head_dim;
            float w = scores[t] / denom;
            for (int d = 0; d < head_dim; d++) o[d] += w * v_t[d];
        }
    }
    return out;
}

// Run one decode attention variant and check against the CPU reference.
// atol=5e-2 because fp16 arithmetic + online softmax introduces small differences.
static void run_decode_attn_test(
    const std::string& name,
    void(*kernel_fn)(const __half*, const __half*, const __half*, __half*, int, int, int, int, cudaStream_t),
    const std::vector<float>& q_h,
    const std::vector<float>& k_h,
    const std::vector<float>& v_h,
    const std::vector<float>& ref,
    int num_heads, int num_kv_heads, int head_dim, int kv_len)
{
    fp16* dq  = upload_fp16(q_h);
    fp16* dkc = upload_fp16(k_h);
    fp16* dvc = upload_fp16(v_h);
    fp16* do_ = nullptr;
    cudaMalloc(&do_, num_heads * head_dim * sizeof(fp16));

    kernel_fn(dq, dkc, dvc, do_, num_heads, num_kv_heads, head_dim, kv_len, nullptr);
    cudaDeviceSynchronize();

    auto out = download_fp16(do_, num_heads * head_dim);
    check(name, allclose(out, ref, 5e-2f));

    cudaFree(dq); cudaFree(dkc); cudaFree(dvc); cudaFree(do_);
}

static void test_decode_attention() {
    // Use a small configuration to keep the test fast and the reference manageable.
    const int num_heads    = 8;
    const int num_kv_heads = 2;  // GQA ratio = 4
    const int head_dim     = 64;
    const int kv_len       = 128;

    auto q  = rand_vec(num_heads    * head_dim, -0.5f, 0.5f);
    auto kc = rand_vec(kv_len * num_kv_heads * head_dim, -0.5f, 0.5f);
    auto vc = rand_vec(kv_len * num_kv_heads * head_dim, -0.5f, 0.5f);

    auto ref = cpu_decode_attn(q, kc, vc, num_heads, num_kv_heads, head_dim, kv_len);

    run_decode_attn_test("decode_attn_v1_coal", decode_attn_v1_coal,
                         q, kc, vc, ref, num_heads, num_kv_heads, head_dim, kv_len);
    run_decode_attn_test("decode_attn_v2_warp_red", decode_attn_v2_warp_red,
                         q, kc, vc, ref, num_heads, num_kv_heads, head_dim, kv_len);
    run_decode_attn_test("decode_attn_v3_vec", decode_attn_v3_vec,
                         q, kc, vc, ref, num_heads, num_kv_heads, head_dim, kv_len);
    run_decode_attn_test("decode_attn_v4_flash", decode_attn_v4_flash,
                         q, kc, vc, ref, num_heads, num_kv_heads, head_dim, kv_len);

    // Also test at kv_len=1 (edge case: just one position in cache)
    {
        const int kv1 = 1;
        auto kc1 = rand_vec(kv1 * num_kv_heads * head_dim, -0.5f, 0.5f);
        auto vc1 = rand_vec(kv1 * num_kv_heads * head_dim, -0.5f, 0.5f);
        auto ref1 = cpu_decode_attn(q, kc1, vc1, num_heads, num_kv_heads, head_dim, kv1);
        run_decode_attn_test("decode_attn_v4_flash (kv_len=1)", decode_attn_v4_flash,
                             q, kc1, vc1, ref1, num_heads, num_kv_heads, head_dim, kv1);
    }

    // Test head_dim=128 (covers the other template instantiation)
    {
        const int hd2 = 128;
        auto q2  = rand_vec(num_heads    * hd2, -0.5f, 0.5f);
        auto kc2 = rand_vec(kv_len * num_kv_heads * hd2, -0.5f, 0.5f);
        auto vc2 = rand_vec(kv_len * num_kv_heads * hd2, -0.5f, 0.5f);
        auto ref2 = cpu_decode_attn(q2, kc2, vc2, num_heads, num_kv_heads, hd2, kv_len);
        run_decode_attn_test("decode_attn_v4_flash (head_dim=128)", decode_attn_v4_flash,
                             q2, kc2, vc2, ref2, num_heads, num_kv_heads, hd2, kv_len);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    srand(12345);

    std::cout << "=== ForgeRT Kernel Correctness Tests ===\n\n";

    test_rmsnorm();
    test_rmsnorm_fused();
    test_swiglu();
    test_residual_add();
    test_rope_tables();
    test_decode_attention();

    std::cout << "\n" << g_tests - g_fails << "/" << g_tests << " tests passed.\n";
    return (g_fails > 0) ? 1 : 0;
}
