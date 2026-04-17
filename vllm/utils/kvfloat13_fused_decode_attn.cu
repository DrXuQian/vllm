/**
 * KVFloat13 Fused Decode Attention Kernel
 *
 * Based on FlashInfer's BatchDecodeWithPagedKVCacheDevice structure,
 * but replaces cp_async K/V loads with in-register KVFloat13 decode.
 *
 * Data flow per thread (vec_size=4):
 *   Global (7 bytes packed) → ldg → registers → decode → BF16 registers
 *   → sts to k_smem/v_smem → compute QK / accumulate V (unchanged)
 *
 * vs BF16 baseline:
 *   Global (8 bytes BF16) → cp_async → k_smem/v_smem → compute
 *
 * HBM savings: 208/256 = 18.75% less traffic per KV head.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================
// KVFloat13 decompress LUT (32 entries, set at init)
// ============================================================
__constant__ uint8_t c_kvf13_decompress[32] = {
    0, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131
};

// KVFloat13 packed layout offsets within a 208-byte chunk (128 values)
constexpr uint32_t KVF13_SIGN_OFF = 0;     // 16 bytes: 128 sign bits
constexpr uint32_t KVF13_EXP_HI_OFF = 16;  // 64 bytes: 128 nibbles
constexpr uint32_t KVF13_EM_OFF = 80;       // 128 bytes: [exp_lo|mant7]
constexpr uint32_t KVF13_CHUNK_BYTES = 208;
constexpr uint32_t KVF13_CHUNK_SIZE = 128;

// ============================================================
// In-register KVFloat13 decode: 7 bytes → 4 BF16 values
// ============================================================

/**
 * Load packed KVFloat13 data from global memory and decode to BF16 in registers.
 *
 * @param packed_chunk  Pointer to the 208-byte chunk in global memory
 * @param pos           Position within the chunk (0..127), must be aligned to 4
 * @param out           Output array of 4 BF16 values
 */
__device__ __forceinline__ void kvf13_decode_4_values(
    const uint8_t* __restrict__ packed_chunk,
    uint32_t pos,       // = tx * 4, aligned to 4
    __nv_bfloat16 out[4]
) {
    // Load sign bits: 1 byte covers 8 positions, we need 4 consecutive bits
    const uint8_t sign_byte = packed_chunk[KVF13_SIGN_OFF + pos / 8];
    const uint32_t sign_shift = pos & 7;  // pos % 8

    // Load exp_hi nibbles: 2 bytes cover 4 positions
    // Nibble packing: byte[i] has low_nibble=even, high_nibble=odd
    const uint8_t eh0 = packed_chunk[KVF13_EXP_HI_OFF + pos / 2];
    const uint8_t eh1 = packed_chunk[KVF13_EXP_HI_OFF + pos / 2 + 1];

    // Load exp_lo_mant: 4 consecutive bytes (coalesced 32-bit load)
    const uint32_t em4 = *reinterpret_cast<const uint32_t*>(
        packed_chunk + KVF13_EM_OFF + pos);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Extract sign bit
        uint32_t sign = (sign_byte >> (sign_shift + i)) & 1u;

        // Extract exp_hi nibble
        uint32_t exp_h4;
        if (i < 2) {
            exp_h4 = (eh0 >> (i * 4)) & 0xFu;
        } else {
            exp_h4 = (eh1 >> ((i - 2) * 4)) & 0xFu;
        }

        // Extract em byte
        uint32_t em_b = (em4 >> (i * 8)) & 0xFFu;

        // Reconstruct 5-bit exponent index
        uint32_t exp5 = (exp_h4 << 1) | (em_b >> 7);

        // Decompress: exp5 → exp8 via constant memory LUT
        uint32_t exp8 = c_kvf13_decompress[exp5];

        // Mantissa (7 bits, untouched)
        uint32_t mant7 = em_b & 0x7Fu;

        // Assemble BF16: sign(1) | exp8(8) | mant7(7)
        uint16_t bf16_bits = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        out[i] = *reinterpret_cast<__nv_bfloat16*>(&bf16_bits);
    }
}

/**
 * Load and decode a tile of K or V values from paged KVFloat13 cache.
 *
 * Replaces cp_async in the original FlashInfer decode kernel.
 * Each thread loads its vec_size (=4) values from the packed page,
 * decodes in registers, and writes BF16 to shared memory.
 *
 * @param smem_dst      Destination in shared memory (BF16)
 * @param packed_cache  KV cache tensor base pointer (uint8)
 * @param kv_idx        KV position index (which token in the sequence)
 * @param kv_head_idx   KV head index
 * @param page_table    Block table for this request
 * @param page_size     Number of tokens per page
 * @param packed_stride_n   Stride in bytes between consecutive KV positions
 * @param packed_stride_h   Stride in bytes between consecutive KV heads
 * @param tx            Thread x index within warp (0..31)
 * @param valid         Whether this position is within sequence bounds
 */
__device__ __forceinline__ void kvf13_load_tile_to_smem(
    __nv_bfloat16* smem_dst,
    const uint8_t* __restrict__ packed_cache,
    uint32_t kv_idx,
    uint32_t kv_head_idx,
    const int32_t* page_table,
    uint32_t page_size,
    uint32_t packed_stride_n,  // stride between tokens
    uint32_t packed_stride_h,  // stride between heads
    uint32_t tx,
    bool valid
) {
    if (!valid) {
        // Fill with zeros for out-of-bounds positions
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            smem_dst[tx * 4 + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Resolve paged address
    uint32_t page_idx = kv_idx / page_size;
    uint32_t page_offset = kv_idx % page_size;
    int32_t physical_page = page_table[page_idx];

    // Calculate pointer to the 208-byte chunk for this (page, offset, head)
    const uint8_t* chunk_ptr = packed_cache
        + (uint64_t)physical_page * page_size * packed_stride_n
        + page_offset * packed_stride_n
        + kv_head_idx * packed_stride_h;

    // Decode 4 values in registers
    __nv_bfloat16 decoded[4];
    kvf13_decode_4_values(chunk_ptr, tx * 4, decoded);

    // Write decoded BF16 to shared memory
    *reinterpret_cast<uint2*>(smem_dst + tx * 4) =
        *reinterpret_cast<uint2*>(decoded);
}


// ============================================================
// Simplified paged decode attention kernel with fused KVFloat13
// ============================================================

/**
 * Batch decode attention with KVFloat13 paged KV cache.
 *
 * Each thread block handles one (batch, kv_head) pair.
 * Within each block, bdy query heads share the same KV head (GQA).
 *
 * Grid:  (num_kv_chunks, num_kv_heads, batch_size)
 * Block: (32, num_qo_heads_per_kv, 1)
 */
template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDY>
__global__ void kvfloat13_batch_decode_attention_kernel(
    const __nv_bfloat16* __restrict__ q,    // [batch, num_qo_heads, head_dim]
    const uint8_t* __restrict__ kv_cache_k,  // [num_pages, page_size, num_kv_heads, packed_bytes]
    const uint8_t* __restrict__ kv_cache_v,  // same layout
    __nv_bfloat16* __restrict__ output,      // [batch, num_qo_heads, head_dim]
    float* __restrict__ lse,                 // [batch, num_qo_heads] (optional)
    const int32_t* __restrict__ page_table,  // [batch, max_num_pages]
    const int32_t* __restrict__ seq_lens,    // [batch]
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t page_size,
    uint32_t max_num_pages,
    float sm_scale,
    uint32_t kv_chunk_size
) {
    constexpr uint32_t BDX = HEAD_DIM / VEC_SIZE;  // = 32 for head_dim=128, vec_size=4

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;  // which qo head within this kv head group
    const uint32_t kv_head_idx = blockIdx.y;
    const uint32_t batch_idx = blockIdx.z;
    const uint32_t kv_chunk_idx = blockIdx.x;
    const uint32_t qo_head_idx = kv_head_idx * BDY + ty;

    const uint32_t seq_len = seq_lens[batch_idx];
    const int32_t* my_page_table = page_table + batch_idx * max_num_pages;

    // Strides for packed KV cache
    const uint32_t packed_bytes_per_head = KVF13_CHUNK_BYTES;  // 208
    const uint32_t packed_stride_h = packed_bytes_per_head;
    const uint32_t packed_stride_n = num_kv_heads * packed_bytes_per_head;

    // Shared memory: k_smem and v_smem, each holds one tile of BF16 values
    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* v_smem = k_smem + BDY * HEAD_DIM;  // after K tile

    // Load query to registers
    float q_reg[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        q_reg[i] = __bfloat162float(
            q[(batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM + tx * VEC_SIZE + i]);
    }

    // Online softmax state
    float m = -1e20f;   // running max
    float d = 0.0f;     // running sum of exp
    float o_reg[VEC_SIZE] = {0};  // running output accumulator

    // Iterate over KV positions in this chunk
    uint32_t chunk_start = kv_chunk_idx * kv_chunk_size;
    uint32_t chunk_end = min(chunk_start + kv_chunk_size, seq_len);

    for (uint32_t kv_pos = chunk_start; kv_pos < chunk_end; kv_pos++) {
        // Load K for this position: packed → registers → BF16 → smem
        bool valid = (kv_pos < seq_len);
        kvf13_load_tile_to_smem(
            k_smem + ty * HEAD_DIM,
            kv_cache_k, kv_pos, kv_head_idx,
            my_page_table, page_size,
            packed_stride_n, packed_stride_h,
            tx, valid
        );
        __syncthreads();

        // Compute QK dot product
        float score = 0.0f;
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            score += q_reg[i] * __bfloat162float(k_smem[ty * HEAD_DIM + tx * VEC_SIZE + i]);
        }
        // Warp reduction
        #pragma unroll
        for (int offset = BDX / 2; offset > 0; offset >>= 1) {
            score += __shfl_xor_sync(0xFFFFFFFF, score, offset);
        }
        score *= sm_scale;
        if (!valid) score = -1e20f;

        // Online softmax update
        float m_prev = m;
        m = fmaxf(m, score);
        float scale = expf(m_prev - m);
        d = d * scale + expf(score - m);

        // Scale previous output accumulator
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            o_reg[i] *= scale;
        }
        __syncthreads();

        // Load V for this position
        kvf13_load_tile_to_smem(
            v_smem + ty * HEAD_DIM,
            kv_cache_v, kv_pos, kv_head_idx,
            my_page_table, page_size,
            packed_stride_n, packed_stride_h,
            tx, valid
        );
        __syncthreads();

        // Accumulate weighted V
        float weight = expf(score - m);
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            o_reg[i] += weight * __bfloat162float(v_smem[ty * HEAD_DIM + tx * VEC_SIZE + i]);
        }
        __syncthreads();
    }

    // Normalize output
    float d_rcp = (d > 0.0f) ? 1.0f / d : 0.0f;
    uint32_t out_offset = (batch_idx * num_qo_heads + qo_head_idx) * HEAD_DIM + tx * VEC_SIZE;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        output[out_offset + i] = __float2bfloat16(o_reg[i] * d_rcp);
    }

    // Store LSE
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
    constexpr uint32_t BDX = HEAD_DIM / VEC_SIZE;  // 32

    uint32_t qo_heads_per_kv = num_qo_heads / num_kv_heads;
    uint32_t kv_chunk_size = 64;  // tokens per chunk
    uint32_t num_kv_chunks = (page_size * max_num_pages + kv_chunk_size - 1) / kv_chunk_size;

    dim3 grid(num_kv_chunks, num_kv_heads, batch_size);
    dim3 block(BDX, qo_heads_per_kv, 1);
    uint32_t smem_size = 2 * qo_heads_per_kv * HEAD_DIM * sizeof(__nv_bfloat16);

    kvfloat13_batch_decode_attention_kernel<HEAD_DIM, VEC_SIZE, 4>
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
