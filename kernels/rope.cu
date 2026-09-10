// ---------------------------------------------------------------------------
// rope.cu
//
// Rotary Position Embedding (RoPE) – applied to Q and K before attention.
//
// For a head dimension d, we split the last dim into d/2 complex pairs:
//   (x_0, x_1), (x_2, x_3), ..., (x_{d-2}, x_{d-1})
//
// Each pair (a, b) at frequency position p is rotated by θ_i:
//   θ_i = p / base^(2i / d)
//   (a', b') = (a·cos(θ) - b·sin(θ),  a·sin(θ) + b·cos(θ))
//
// We precompute cos/sin tables on the GPU once at startup.
//
// Optimization progression:
//   v0 – scalar access, recompute sin/cos per thread
//   v1 – coalesced: each thread handles exactly one (a,b) pair from the table
//   v2 – fused Q+K in one kernel, halving kernel launch overhead
// ---------------------------------------------------------------------------
#include "rope.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Precompute tables
// ---------------------------------------------------------------------------
__global__ void compute_rope_tables_kernel(
    float* __restrict__ cos_tab,   // [max_seq_len, head_dim/2]
    float* __restrict__ sin_tab,
    int max_seq_len, int head_dim, float base, float scale)
{
    int pos  = blockIdx.x;
    int freq = threadIdx.x;        // 0 .. head_dim/2 - 1
    int half_d = head_dim / 2;

    if (freq >= half_d) return;

    float theta = (float)pos / (powf(base, 2.0f * freq / (float)head_dim) * scale);
    cos_tab[pos * half_d + freq] = cosf(theta);
    sin_tab[pos * half_d + freq] = sinf(theta);
}

void rope_precompute_tables(
    float** cos_out, float** sin_out,
    int max_seq_len, int head_dim, float base, float scale)
{
    size_t table_size = (size_t)max_seq_len * (head_dim / 2) * sizeof(float);
    cudaMalloc(cos_out, table_size);
    cudaMalloc(sin_out, table_size);

    dim3 grid(max_seq_len);
    dim3 block(head_dim / 2);
    compute_rope_tables_kernel<<<grid, block>>>(*cos_out, *sin_out, max_seq_len, head_dim, base, scale);
    cudaDeviceSynchronize();
}

// ---------------------------------------------------------------------------
// Helper: apply RoPE to one tensor [seq_len, num_heads, head_dim] in-place
// ---------------------------------------------------------------------------

// v0 – naive: thread = one element of the frequency pair
__global__ void rope_kernel_v0(
    __half* __restrict__ x,            // [seq_len, num_heads, head_dim]
    const float* __restrict__ cos_tab, // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_tab,
    int seq_len, int num_heads, int head_dim, int pos_offset)
{
    // tid encodes (tok, head, freq_pair)
    int freq   = threadIdx.x;
    int head   = blockIdx.x % num_heads;
    int tok    = blockIdx.x / num_heads;
    int half_d = head_dim / 2;

    if (freq >= half_d || tok >= seq_len) return;

    int pos   = tok + pos_offset;
    float c   = cos_tab[pos * half_d + freq];
    float s   = sin_tab[pos * half_d + freq];

    // Pair (freq, freq + half_d)
    int base_idx = (tok * num_heads + head) * head_dim;
    float a = __half2float(x[base_idx + freq]);
    float b = __half2float(x[base_idx + freq + half_d]);

    x[base_idx + freq]          = __float2half(a * c - b * s);
    x[base_idx + freq + half_d] = __float2half(a * s + b * c);
}

void rope_v0_naive(
    __half* q, __half* k,
    const float* cos_tab, const float* sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset, cudaStream_t stream)
{
    int half_d = head_dim / 2;
    // Q
    rope_kernel_v0<<<seq_len * num_q_heads, half_d, 0, stream>>>(
        q, cos_tab, sin_tab, seq_len, num_q_heads, head_dim, pos_offset);
    // K
    rope_kernel_v0<<<seq_len * num_kv_heads, half_d, 0, stream>>>(
        k, cos_tab, sin_tab, seq_len, num_kv_heads, head_dim, pos_offset);
}

// ---------------------------------------------------------------------------
// v1 – coalesced: one block = (token, head), threads = frequency pairs
// Same kernel can handle both Q and K.
// ---------------------------------------------------------------------------
__global__ void rope_kernel_v1_coal(
    __half* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int num_heads, int head_dim, int pos_offset)
{
    int half_d = head_dim / 2;
    int tok    = blockIdx.y;
    int head   = blockIdx.x;

    int pos    = tok + pos_offset;

    // threads handle one pair each
    for (int freq = threadIdx.x; freq < half_d; freq += blockDim.x) {
        float c = cos_tab[pos * half_d + freq];
        float s = sin_tab[pos * half_d + freq];

        int base_idx = (tok * num_heads + head) * head_dim;
        float a = __half2float(x[base_idx + freq]);
        float b = __half2float(x[base_idx + freq + half_d]);

        x[base_idx + freq]          = __float2half(a * c - b * s);
        x[base_idx + freq + half_d] = __float2half(a * s + b * c);
    }
}

void rope_v1_coal(
    __half* q, __half* k,
    const float* cos_tab, const float* sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset, cudaStream_t stream)
{
    int half_d = head_dim / 2;
    int threads = min(half_d, 128);

    // Q
    {
        dim3 grid(num_q_heads, seq_len);
        rope_kernel_v1_coal<<<grid, threads, 0, stream>>>(
            q, cos_tab, sin_tab, num_q_heads, head_dim, pos_offset);
    }
    // K
    {
        dim3 grid(num_kv_heads, seq_len);
        rope_kernel_v1_coal<<<grid, threads, 0, stream>>>(
            k, cos_tab, sin_tab, num_kv_heads, head_dim, pos_offset);
    }
}

// ---------------------------------------------------------------------------
// v2 – fused Q+K in one kernel
// blockIdx.z == 0 → Q tensor,  blockIdx.z == 1 → K tensor
// ---------------------------------------------------------------------------
__global__ void rope_kernel_v2_fused(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    int num_q_heads, int num_kv_heads, int head_dim, int pos_offset)
{
    int half_d = head_dim / 2;
    int is_k   = blockIdx.z;   // 0=Q, 1=K
    int tok    = blockIdx.y;
    int head   = blockIdx.x;

    int num_heads = is_k ? num_kv_heads : num_q_heads;
    if (head >= num_heads) return;

    __half* x = is_k ? k : q;
    int pos   = tok + pos_offset;

    for (int freq = threadIdx.x; freq < half_d; freq += blockDim.x) {
        float c = cos_tab[pos * half_d + freq];
        float s = sin_tab[pos * half_d + freq];

        int base_idx = (tok * num_heads + head) * head_dim;
        float a = __half2float(x[base_idx + freq]);
        float b = __half2float(x[base_idx + freq + half_d]);

        x[base_idx + freq]          = __float2half(a * c - b * s);
        x[base_idx + freq + half_d] = __float2half(a * s + b * c);
    }
}

void rope_v2_fused(
    __half* q, __half* k,
    const float* cos_tab, const float* sin_tab,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    int pos_offset, cudaStream_t stream)
{
    int half_d  = head_dim / 2;
    int threads = min(half_d, 128);
    int max_heads = max(num_q_heads, num_kv_heads);

    // z=0 for Q, z=1 for K; head dim covers both (K heads will short-circuit)
    dim3 grid(max_heads, seq_len, 2);
    rope_kernel_v2_fused<<<grid, threads, 0, stream>>>(
        q, k, cos_tab, sin_tab, num_q_heads, num_kv_heads, head_dim, pos_offset);
}
