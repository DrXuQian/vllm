#ifndef FLASHINFER_DECODE_KVFLOAT13_CUH_
#define FLASHINFER_DECODE_KVFLOAT13_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define KVF13_SIGN_OFF   0
#define KVF13_EXP_HI_OFF 16
#define KVF13_EM_OFF     80
#define KVF13_CHUNK_BYTES 208

namespace flashinfer {

template <typename DTypeKV, uint32_t HEAD_DIM>
struct KVSmemStride {
    static constexpr uint32_t value = HEAD_DIM * sizeof(DTypeKV);
};
template <uint32_t HEAD_DIM>
struct KVSmemStride<uint8_t, HEAD_DIM> {
    static constexpr uint32_t value = KVF13_CHUNK_BYTES;
};

template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_decode_vec(
    vec_t<float, vec_size>& out,
    const uint8_t* chunk,
    uint32_t start_idx
) {
    static_assert(vec_size == 8, "kvf13_decode_vec requires vec_size=8");
    const uint32_t tx = start_idx >> 3;
    const uint32_t sign_byte = chunk[tx];
    const uint32_t exp_hi_w  = *reinterpret_cast<const uint32_t*>(chunk + 16 + tx * 4);
    const uint2    em_pair   = *reinterpret_cast<const uint2*>(chunk + 80 + tx * 8);

    #pragma unroll
    for (uint32_t p = 0; p < 4; p++) {
        const uint32_t k0 = 2u * p;
        const uint32_t k1 = k0 + 1u;
        const uint32_t s0 = (sign_byte >> k0) & 1u;
        const uint32_t s1 = (sign_byte >> k1) & 1u;
        const uint32_t eh_byte = (exp_hi_w >> (8u * p)) & 0xFFu;
        const uint32_t n0 = eh_byte & 0xFu;
        const uint32_t n1 = (eh_byte >> 4u) & 0xFu;
        const uint32_t em_word = (p < 2) ? em_pair.x : em_pair.y;
        const uint32_t em0 = (em_word >> (8u * (k0 & 3u))) & 0xFFu;
        const uint32_t em1 = (em_word >> (8u * (k1 & 3u))) & 0xFFu;
        const uint32_t nz0 = n0 | (em0 >> 7u);
        const uint32_t nz1 = n1 | (em1 >> 7u);
        const uint32_t lo0 = (n0 << 8u) + em0 + (12800u & -(nz0 != 0u));
        const uint32_t lo1 = (n1 << 8u) + em1 + (12800u & -(nz1 != 0u));
        uint16_t bf0 = static_cast<uint16_t>(lo0);
        uint16_t bf1 = static_cast<uint16_t>(lo1);
        bf0 |= static_cast<uint16_t>(s0 << 15u);
        bf1 |= static_cast<uint16_t>(s1 << 15u);
        out[k0] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bf0));
        out[k1] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bf1));
    }

}

template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_decode_vec_bf16(
    __nv_bfloat16* out,
    const uint8_t* chunk,
    uint32_t start_idx
) {
    static_assert(vec_size == 8, "kvf13_decode_vec_bf16 requires vec_size=8");
    const uint32_t tx = start_idx >> 3;
    const uint32_t sign_byte = chunk[tx];
    const uint32_t exp_hi_w  = *reinterpret_cast<const uint32_t*>(chunk + 16 + tx * 4);
    const uint2    em_pair   = *reinterpret_cast<const uint2*>(chunk + 80 + tx * 8);

    #pragma unroll
    for (uint32_t p = 0; p < 4; ++p) {
        const uint32_t k0 = 2u * p;
        const uint32_t k1 = k0 + 1u;
        const uint32_t s0 = (sign_byte >> k0) & 1u;
        const uint32_t s1 = (sign_byte >> k1) & 1u;
        const uint32_t eh_byte = (exp_hi_w >> (8u * p)) & 0xFFu;
        const uint32_t n0 = eh_byte & 0xFu;
        const uint32_t n1 = (eh_byte >> 4u) & 0xFu;
        const uint32_t em_word = (p < 2) ? em_pair.x : em_pair.y;
        const uint32_t em0 = (em_word >> (8u * (k0 & 3u))) & 0xFFu;
        const uint32_t em1 = (em_word >> (8u * (k1 & 3u))) & 0xFFu;
        const uint32_t nz0 = n0 | (em0 >> 7u);
        const uint32_t nz1 = n1 | (em1 >> 7u);
        const uint32_t lo0 = (n0 << 8u) + em0 + (12800u & -(nz0 != 0u));
        const uint32_t lo1 = (n1 << 8u) + em1 + (12800u & -(nz1 != 0u));
        const uint16_t bf0 = static_cast<uint16_t>((s0 << 15u) | lo0);
        const uint16_t bf1 = static_cast<uint16_t>((s1 << 15u) | lo1);
        out[k0] = *reinterpret_cast<const __nv_bfloat16*>(&bf0);
        out[k1] = *reinterpret_cast<const __nv_bfloat16*>(&bf1);
    }
}

template <uint32_t vec_size, uint32_t tile_size_per_bdx, uint32_t bdy, uint32_t head_dim>
__device__ __forceinline__ void kvf13_decode_tiles_to_bf16(
    const uint8_t* packed_smem,
    __nv_bfloat16* decoded_smem,
    uint32_t tx,
    uint32_t ty
) {
    constexpr uint32_t tile_size = tile_size_per_bdx * bdy;
    #pragma unroll
    for (uint32_t j = 0; j < tile_size_per_bdx; ++j) {
        const uint32_t tile_idx = ty * tile_size_per_bdx + j;
        kvf13_decode_vec_bf16<vec_size>(
            decoded_smem + tile_idx * head_dim + tx * vec_size,
            packed_smem + tile_idx * KVF13_CHUNK_BYTES,
            tx * vec_size);
    }
}

__device__ __forceinline__ void kvf13_cp_async_chunk(
    uint8_t* staging_smem,
    const uint8_t* global_chunk,
    uint32_t tx,
    bool valid
) {
    constexpr uint32_t NUM_LOADS = KVF13_CHUNK_BYTES / 16;
    bool active = valid && (tx < NUM_LOADS);
    if (active) {
        uint32_t smem_addr = __cvta_generic_to_shared(staging_smem + tx * 16);
        const void* gmem_ptr = global_chunk + tx * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem_addr), "l"(gmem_ptr));
    }
}

static constexpr int KVF13_LAUNCH_BOUNDS_MIN_BLOCKS = 2;

}  // namespace flashinfer
#endif
