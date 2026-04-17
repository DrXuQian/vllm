#ifndef FLASHINFER_DECODE_KVFLOAT13_CUH_
#define FLASHINFER_DECODE_KVFLOAT13_CUH_

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define KVF13_SIGN_OFF   0
#define KVF13_EXP_HI_OFF 16
#define KVF13_EM_OFF     80
#define KVF13_CHUNK_BYTES 208

namespace flashinfer {

// smem stride per KV position: 208B for KVFloat13, head_dim*sizeof(T) for others
template <typename DTypeKV, uint32_t HEAD_DIM>
struct KVSmemStride {
    static constexpr uint32_t value = HEAD_DIM * sizeof(DTypeKV);
};
template <uint32_t HEAD_DIM>
struct KVSmemStride<uint8_t, HEAD_DIM> {
    static constexpr uint32_t value = KVF13_CHUNK_BYTES;  // 208
};

// Decode vec_size(=8) values from a packed KVFloat13 chunk in smem.
// 3 smem loads: sign(1B) + exp_hi(4B) + em(8B), no LUT, minimal ALU.
//
// Layout (208B per 128 values):
//   [0..15]   signs:  128 bits = 16 bytes
//   [16..79]  exp_hi: 128 nibbles = 64 bytes
//   [80..207] em:     128 bytes (exp_lo1:1 + mant7:7)
template <uint32_t vec_size>
__device__ __forceinline__ void kvf13_decode_vec(
    vec_t<float, vec_size>& out,
    const uint8_t* chunk,       // 208B packed chunk base in smem
    uint32_t start_idx          // value index within head (0..127), must be multiple of 8
) {
    static_assert(vec_size == 8, "kvf13_decode_vec requires vec_size=8");
    const uint32_t tx = start_idx >> 3;

    // --- 3 wide smem loads ---
    const uint32_t sign_byte = chunk[tx];                                                   // lds.u8
    const uint32_t exp_hi_w  = *reinterpret_cast<const uint32_t*>(chunk + 16 + tx * 4);     // lds.32
    const uint2    em_pair   = *reinterpret_cast<const uint2*>(chunk + 80 + tx * 8);         // lds.64

    // Paired decode: process 2 elements per iteration, branchless
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
        const uint16_t bf0 = (uint16_t)((s0 << 15u) | lo0);
        const uint16_t bf1 = (uint16_t)((s1 << 15u) | lo1);
        out[k0] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bf0));
        out[k1] = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&bf1));
    }
}

// cp_async load 208B packed chunk: 13 threads × 16B
__device__ __forceinline__ void kvf13_cp_async_chunk(
    uint8_t* staging_smem,
    const uint8_t* global_chunk,
    uint32_t tx,
    bool valid
) {
    constexpr uint32_t NUM_LOADS = KVF13_CHUNK_BYTES / 16;  // 13
    bool active = valid && (tx < NUM_LOADS);
    if (active) {
        uint32_t smem_addr = __cvta_generic_to_shared(staging_smem + tx * 16);
        const void* gmem_ptr = global_chunk + tx * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem_addr), "l"(gmem_ptr));
    }
}

// Minimum blocks per SM for __launch_bounds__ on KVF13 decode kernel.
// block_size=128 (bdx*bdy*bdz), MIN_BLOCKS=10 → max 51 regs/thread → ~25% more occupancy.
static constexpr int KVF13_LAUNCH_BOUNDS_MIN_BLOCKS = 10;

}  // namespace flashinfer
#endif
