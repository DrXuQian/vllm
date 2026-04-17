/*
 * KVFloat13 in-register decode for FlashInfer decode kernels.
 *
 * Two-phase approach:
 * Phase 1: cp_async load 208B packed chunk to staging smem (coalesced, async)
 * Phase 2: decode from staging smem to BF16 in k_smem/v_smem (fast, parallel)
 */
#ifndef FLASHINFER_DECODE_KVFLOAT13_CUH_
#define FLASHINFER_DECODE_KVFLOAT13_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "../cp_async.cuh"

#define KVF13_SIGN_OFF   0
#define KVF13_EXP_HI_OFF 16
#define KVF13_EM_OFF     80
#define KVF13_CHUNK_BYTES 208
#define KVF13_CP_ASYNC_LOADS 13  // 208 / 16

__constant__ uint8_t d_kvf13_lut[32] = {
    0, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115,
    116, 117, 118, 119, 120, 121, 122, 123,
    124, 125, 126, 127, 128, 129, 130, 131
};

// Phase 1: cp_async 208B packed chunk → staging smem
// 13 threads × 16B cp_async<128>. Thread >= 13 predicated off.
__device__ __forceinline__ void kvf13_cp_async_to_staging(
    uint8_t* staging_smem,
    const uint8_t* global_chunk,
    uint32_t tx,
    bool valid
) {
    flashinfer::cp_async::pred_load<128, flashinfer::cp_async::PrefetchMode::kPrefetch, flashinfer::cp_async::SharedMemFillMode::kFillZero>(
        staging_smem + tx * 16,
        global_chunk + tx * 16,
        valid && tx < KVF13_CP_ASYNC_LOADS
    );
}

// Phase 2: decode staged packed smem → BF16 smem
template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_decode_staged(
    __nv_bfloat16* bf16_dst,
    const uint8_t* staging_src,
    uint32_t pos_in_head
) {
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        uint32_t pos = pos_in_head + i;
        uint32_t sign = (staging_src[KVF13_SIGN_OFF + pos / 8] >> (pos & 7)) & 1u;
        uint32_t eh = staging_src[KVF13_EXP_HI_OFF + pos / 2];
        uint32_t exp_h4 = (pos & 1) ? ((eh >> 4) & 0xFu) : (eh & 0xFu);
        uint32_t em = staging_src[KVF13_EM_OFF + pos];
        uint32_t exp5 = (exp_h4 << 1) | (em >> 7);
        uint32_t exp8 = d_kvf13_lut[exp5];
        uint32_t mant7 = em & 0x7Fu;
        uint16_t bf16 = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        bf16_dst[pos] = *reinterpret_cast<__nv_bfloat16*>(&bf16);
    }
}

// Fallback: direct ldg load + decode (for SingleDecode / non-pipelined use)
template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_load_decode_store(
    __nv_bfloat16* smem_dst,
    const uint8_t* chunk,
    uint32_t pos_in_head,
    bool pred
) {
    if (!pred) {
        #pragma unroll
        for (uint32_t i = 0; i < vec_size; ++i)
            smem_dst[pos_in_head + i] = __float2bfloat16(0.0f);
        return;
    }
    #pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
        uint32_t pos = pos_in_head + i;
        uint32_t sign = (chunk[KVF13_SIGN_OFF + pos / 8] >> (pos & 7)) & 1u;
        uint32_t eh = chunk[KVF13_EXP_HI_OFF + pos / 2];
        uint32_t exp_h4 = (pos & 1) ? ((eh >> 4) & 0xFu) : (eh & 0xFu);
        uint32_t em = chunk[KVF13_EM_OFF + pos];
        uint32_t exp5 = (exp_h4 << 1) | (em >> 7);
        uint32_t exp8 = d_kvf13_lut[exp5];
        uint32_t mant7 = em & 0x7Fu;
        uint16_t bf16 = (uint16_t)((sign << 15) | (exp8 << 7) | mant7);
        smem_dst[pos] = *reinterpret_cast<__nv_bfloat16*>(&bf16);
    }
}

#endif
