// ---------------------------------------------------------------------------
// embed.cu – Token embedding gather
// ---------------------------------------------------------------------------
#include "embed.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// One block per token; threads stride over hidden_size.
__global__ void embed_gather_kernel(
    const __half*  __restrict__ table,
    const int32_t* __restrict__ ids,
    __half*        __restrict__ out,
    int hidden_size)
{
    int tok = blockIdx.x;
    int row = ids[tok];
    const __half* src = table + (size_t)row * hidden_size;
    __half*       dst = out   + (size_t)tok * hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x)
        dst[i] = src[i];
}

void embed_gather(
    const __half* table, const int32_t* ids, __half* out,
    int seq_len, int hidden_size, cudaStream_t stream)
{
    int threads = min(hidden_size, 256);
    embed_gather_kernel<<<seq_len, threads, 0, stream>>>(table, ids, out, hidden_size);
}
