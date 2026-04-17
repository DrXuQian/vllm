/**
 * KVFloat13 Fused Decode Attention Kernel (v2 — tile batching)
 *
 * Reads packed KVFloat13 directly from paged KV cache,
 * decodes to BF16 in registers, computes attention.
 *
 * v2: Process TILE_SIZE KV positions per iteration for better
 * instruction-level parallelism and reduced sync overhead.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// KVFloat13 decompress LUT
__constant__ uint8_t c_kvf13_decompress[32] = {
    0, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131
};

constexpr uint32_t KVF13_SIGN_OFF = 0;
constexpr uint32_t KVF13_EXP_HI_OFF = 16;
constexpr uint32_t KVF13_EM_OFF = 80;
constexpr uint32_t KVF13_CHUNK_BYTES = 208;

// ============================================================
// In-register KVFloat13 decode
// ============================================================

__device__ __forceinline__ void kvf13_decode_4(
    const uint8_t* __restrict__ chunk,
    uint32_t pos,
    __nv_bfloat16 out[4]
) {
    const uint8_t sign_byte = chunk[KVF13_SIGN_OFF + pos / 8];
    const uint32_t sign_shift = pos & 7;
    const uint8_t eh0 = chunk[KVF13_EXP_HI_OFF + pos / 2];
    const uint8_t eh1 = chunk[KVF13_EXP_HI_OFF + pos / 2 + 1];
    const uint32_t em4 = *reinterpret_cast<const uint32_t*>(chunk + KVF13_EM_OFF + pos);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t sign = (sign_byte >> (sign_shift + i)) & 1u;
        uint32_t exp_h4 = (i < 2) ? ((eh0 >> (i * 4)) & 0xFu)
                                   : ((eh1 >> ((i - 2) * 4)) & 0xFu);
        uint32_t em_b = (em4 >> (i * 8)) & 0xFFu;
        uint32_t exp5 = (exp_h4 << 1) | (em_b >> 7);
        uint32_t exp8 = c_kvf13_decompress[exp5];
        uint32_t mant7 = em_b & 0x7Fu;
        uint16_t bf16 = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        out[i] = *reinterpret_cast<__nv_bfloat16*>(&bf16);
    }
}

__device__ __forceinline__ void kvf13_load_to_smem(
    __nv_bfloat16* dst,
    const uint8_t* __restrict__ cache,
    uint32_t kv_idx,
    uint32_t kv_head_idx,
    const int32_t* page_table,
    uint32_t page_size,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t tx,
    bool valid
) {
    if (!valid) {
        *reinterpret_cast<uint2*>(dst + tx * 4) = make_uint2(0, 0);
        return;
    }
    uint32_t page_idx = kv_idx / page_size;
    uint32_t page_off = kv_idx % page_size;
    int32_t phys_page = page_table[page_idx];
    const uint8_t* chunk = cache + (uint64_t)phys_page * page_size * stride_n
                         + page_off * stride_n + kv_head_idx * stride_h;

    __nv_bfloat16 decoded[4];
    kvf13_decode_4(chunk, tx * 4, decoded);
    *reinterpret_cast<uint2*>(dst + tx * 4) = *reinterpret_cast<uint2*>(decoded);
}

// ============================================================
// Tiled decode attention kernel
// ============================================================

template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDY, uint32_t TILE_SIZE>
__global__ void kvfloat13_batch_decode_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv_cache_k,
    const uint8_t* __restrict__ kv_cache_v,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ lse,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t page_size,
    uint32_t max_num_pages,
    float sm_scale,
    uint32_t kv_chunk_size
) {
    constexpr uint32_t BDX = HEAD_DIM / VEC_SIZE;

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t kv_head_idx = blockIdx.y;
    const uint32_t batch_idx = blockIdx.z;
    const uint32_t kv_chunk_idx = blockIdx.x;
    const uint32_t qo_head_idx = kv_head_idx * BDY + ty;

    const uint32_t seq_len = seq_lens[batch_idx];
    const int32_t* my_page_table = page_table + batch_idx * max_num_pages;
    const uint32_t stride_h = KVF13_CHUNK_BYTES;
    const uint32_t stride_n = num_kv_heads * KVF13_CHUNK_BYTES;

    // Shared memory: TILE_SIZE rows of K, then TILE_SIZE rows of V
    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* v_smem = k_smem + TILE_SIZE * HEAD_DIM;

    // Query in registers
    float q_reg[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_reg[i] = __bfloat162float(
            q[(batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM + tx * VEC_SIZE + i]);
    }

    // Online softmax state
    float m = -1e20f;
    float d = 0.0f;
    float o_reg[VEC_SIZE] = {0};

    uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
    uint32_t chunk_end = min(chunk_start + kv_chunk_size, seq_len);

    // Process TILE_SIZE positions per iteration
    for (uint32_t tile_start = chunk_start; tile_start < chunk_end; tile_start += TILE_SIZE) {
        uint32_t tile_end = min(tile_start + TILE_SIZE, chunk_end);
        uint32_t tile_len = tile_end - tile_start;

        // ---- Load K tile: TILE_SIZE positions ----
        // Each thread loads its vec_size values for ONE position
        // ty selects which position within the tile this thread handles
        // We iterate if TILE_SIZE > BDY
        #pragma unroll
        for (uint32_t t = ty; t < TILE_SIZE; t += BDY) {
            uint32_t kv_pos = tile_start + t;
            bool valid = (kv_pos < chunk_end);
            kvf13_load_to_smem(
                k_smem + t * HEAD_DIM,
                kv_cache_k, kv_pos, kv_head_idx,
                my_page_table, page_size, stride_n, stride_h,
                tx, valid
            );
        }
        __syncthreads();

        // ---- Compute QK for all tile positions ----
        float s[TILE_SIZE];
        #pragma unroll
        for (uint32_t t = 0; t < TILE_SIZE; t++) {
            s[t] = 0.0f;
            if (tile_start + t < chunk_end) {
                #pragma unroll
                for (int i = 0; i < VEC_SIZE; i++) {
                    s[t] += q_reg[i] * __bfloat162float(k_smem[t * HEAD_DIM + tx * VEC_SIZE + i]);
                }
                // Warp reduction
                #pragma unroll
                for (uint32_t offset = BDX / 2; offset > 0; offset >>= 1) {
                    s[t] += __shfl_xor_sync(0xFFFFFFFF, s[t], offset);
                }
                s[t] *= sm_scale;
            } else {
                s[t] = -1e20f;
            }
        }

        // ---- Online softmax update for entire tile ----
        float m_prev = m;
        #pragma unroll
        for (uint32_t t = 0; t < TILE_SIZE; t++) {
            m = fmaxf(m, s[t]);
        }
        float rescale = expf(m_prev - m);
        d *= rescale;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            o_reg[i] *= rescale;
        }
        #pragma unroll
        for (uint32_t t = 0; t < TILE_SIZE; t++) {
            s[t] = expf(s[t] - m);
            d += s[t];
        }
        __syncthreads();

        // ---- Load V tile ----
        #pragma unroll
        for (uint32_t t = ty; t < TILE_SIZE; t += BDY) {
            uint32_t kv_pos = tile_start + t;
            bool valid = (kv_pos < chunk_end);
            kvf13_load_to_smem(
                v_smem + t * HEAD_DIM,
                kv_cache_v, kv_pos, kv_head_idx,
                my_page_table, page_size, stride_n, stride_h,
                tx, valid
            );
        }
        __syncthreads();

        // ---- Accumulate weighted V ----
        #pragma unroll
        for (uint32_t t = 0; t < TILE_SIZE; t++) {
            #pragma unroll
            for (int i = 0; i < VEC_SIZE; i++) {
                o_reg[i] += s[t] * __bfloat162float(v_smem[t * HEAD_DIM + tx * VEC_SIZE + i]);
            }
        }
        __syncthreads();
    }

    // Normalize and write output
    float d_rcp = (d > 0.0f) ? 1.0f / d : 0.0f;
    uint32_t out_off = (batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM + tx * VEC_SIZE;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        output[out_off + i] = __float2bfloat16(o_reg[i] * d_rcp);
    }
    if (lse != nullptr && tx == 0) {
        lse[batch_idx * num_qo_heads + qo_head_idx] = m + logf(d);
    }
}

// ============================================================
// Host launcher
// ============================================================
extern "C" void kvfloat13_fused_decode_attention(
    const void* q,
    const void* kv_cache_k,
    const void* kv_cache_v,
    void* output,
    float* lse,
    const int32_t* page_table,
    const int32_t* seq_lens,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t max_num_pages,
    float sm_scale,
    cudaStream_t stream
) {
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t VEC_SIZE = 4;
    constexpr uint32_t BDX = HEAD_DIM / VEC_SIZE;
    constexpr uint32_t TILE_SIZE = 4;  // KV positions per tile

    uint32_t qo_heads_per_kv = num_qo_heads / num_kv_heads;
    uint32_t kv_chunk_size = 256;
    // Use max_num_pages as a proxy for max_seq_len
    // Caller should pass actual max_seq_len via max_num_pages parameter
    uint32_t num_kv_chunks = (max_num_pages + kv_chunk_size - 1) / kv_chunk_size;
    if (num_kv_chunks == 0) num_kv_chunks = 1;

    dim3 grid(num_kv_chunks, num_kv_heads, batch_size);
    dim3 block(BDX, qo_heads_per_kv, 1);
    // smem: TILE_SIZE rows of K + TILE_SIZE rows of V
    uint32_t smem_size = 2 * TILE_SIZE * HEAD_DIM * sizeof(__nv_bfloat16);

    kvfloat13_batch_decode_attention_kernel<HEAD_DIM, VEC_SIZE, 4, TILE_SIZE>
        <<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q),
        reinterpret_cast<const uint8_t*>(kv_cache_k),
        reinterpret_cast<const uint8_t*>(kv_cache_v),
        reinterpret_cast<__nv_bfloat16*>(output),
        lse,
        page_table,
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        page_size,
        max_num_pages,
        sm_scale,
        kv_chunk_size
    );
}
