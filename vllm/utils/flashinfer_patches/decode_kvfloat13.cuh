/*
 * KVFloat13 fused decode for FlashInfer's BatchDecodeWithPagedKVCache.
 *
 * Replaces cp_async K/V loads with: ldg packed → decode in registers → sts BF16 to smem.
 * All compute (QK, softmax, V accumulate) remains unchanged.
 *
 * Usage: #define FLASHINFER_KVFLOAT13 before including decode.cuh,
 *        or call this function directly.
 */
#ifndef FLASHINFER_DECODE_KVFLOAT13_CUH_
#define FLASHINFER_DECODE_KVFLOAT13_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// KVFloat13 layout constants
#define KVF13_SIGN_OFF   0
#define KVF13_EXP_HI_OFF 16
#define KVF13_EM_OFF     80
#define KVF13_CHUNK_BYTES 208

// Decompress LUT — must be set by host before kernel launch
__constant__ uint8_t d_kvf13_lut[32] = {
    0, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131
};

/*!
 * \brief Load vec_size packed KVFloat13 values from global memory,
 *        decode to BF16 in registers, and store to shared memory.
 *
 * Replaces: cp_async::pred_load<vec_bits>(smem_dst, kv_data + offset, pred)
 *
 * \param smem_dst  Destination in shared memory (BF16)
 * \param kv_data   Global KV cache pointer (uint8_t, packed KVFloat13)
 * \param kv_offset Offset in ELEMENTS (not bytes) — same as original kv_offset
 *                  but scaled for packed format
 * \param pred      Whether this load is valid (in bounds)
 * \param tx        Thread x index (determines position within head_dim)
 * \param vec_size  Number of values per thread (typically 4)
 * \param head_dim  Original head dimension (128)
 */
template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_load_decode_store(
    __nv_bfloat16* smem_dst,  // destination in shared memory
    const uint8_t* chunk,     // pointer to start of 208-byte chunk
    uint32_t pos_in_head,     // position within head (tx * vec_size), 0-aligned
    bool pred                 // validity predicate
) {
    if (!pred) {
        // Zero fill
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i) {
            smem_dst[i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Load sign bits: vec_size bits starting at pos_in_head
    // For vec_size=8: exactly 1 byte. For vec_size=4: half a byte.
    const uint32_t sign_byte_idx = pos_in_head / 8;
    const uint32_t sign_bit_off = pos_in_head & 7;

    // Load exp_hi nibbles: vec_size/2 bytes
    const uint32_t eh_byte_idx = pos_in_head / 2;

    // Load exp_lo_mant: vec_size bytes
    const uint32_t em_byte_idx = pos_in_head;

    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        // Sign: 1 bit
        uint32_t sign = (chunk[KVF13_SIGN_OFF + (pos_in_head + i) / 8] >> ((pos_in_head + i) & 7)) & 1u;

        // Exp_hi: 1 nibble (4 bits)
        uint32_t eh_byte = chunk[KVF13_EXP_HI_OFF + (pos_in_head + i) / 2];
        uint32_t exp_h4 = ((pos_in_head + i) & 1) ? ((eh_byte >> 4) & 0xFu) : (eh_byte & 0xFu);

        // Exp_lo + mantissa: 1 byte
        uint32_t em_b = chunk[KVF13_EM_OFF + pos_in_head + i];

        // Reconstruct
        uint32_t exp5 = (exp_h4 << 1) | (em_b >> 7);
        uint32_t exp8 = d_kvf13_lut[exp5];
        uint32_t mant7 = em_b & 0x7Fu;
        uint16_t bf16 = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        smem_dst[i] = *reinterpret_cast<__nv_bfloat16*>(&bf16);
    }
}

/*!
 * \brief Compute the byte offset for a KVFloat13 packed chunk.
 *
 * For NHD layout: offset = page_idx * stride_page + entry_idx * stride_n + head_idx * stride_h
 * where strides are in BYTES (packed_bytes_per_head = 208, not head_dim = 128).
 *
 * \param page_idx   Physical page index
 * \param head_idx   KV head index
 * \param entry_idx  Entry within page (token position in page)
 * \param num_heads  Number of KV heads
 * \param page_size  Tokens per page
 * \return Byte offset from kv_data start
 */
__device__ __forceinline__ size_t kvf13_get_chunk_offset(
    uint32_t page_idx,
    uint32_t head_idx,
    uint32_t entry_idx,
    uint32_t num_heads,
    uint32_t page_size
) {
    // NHD: [num_pages, page_size, num_heads, 208]
    return (size_t)page_idx * page_size * num_heads * KVF13_CHUNK_BYTES
         + entry_idx * num_heads * KVF13_CHUNK_BYTES
         + head_idx * KVF13_CHUNK_BYTES;
}

#endif  // FLASHINFER_DECODE_KVFLOAT13_CUH_
