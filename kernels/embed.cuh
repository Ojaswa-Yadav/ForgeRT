#pragma once
// ---------------------------------------------------------------------------
// embed.cuh – Token embedding gather kernel
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cstdint>

// Gather embedding rows: out[t, :] = table[ids[t], :]
// table: [vocab_size, hidden_size]  fp16
// ids:   [seq_len]                  int32, on device
// out:   [seq_len, hidden_size]     fp16, on device
void embed_gather(
    const __half*   __restrict__ table,
    const int32_t*  __restrict__ ids,
    __half*         __restrict__ out,
    int seq_len, int hidden_size,
    cudaStream_t stream = nullptr);
