#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>

namespace {

constexpr int kKVFloat13ChunkSize = 128;
constexpr int kKVFloat13SignBytes = 16;
constexpr int kKVFloat13ExpHiBytes = 64;
constexpr int kKVFloat13ChunkBytes =
    kKVFloat13SignBytes + kKVFloat13ExpHiBytes + kKVFloat13ChunkSize;

template <typename index_t>
__global__ void reshape_and_cache_kvfloat13_bf16_kernel(
    const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const index_t* __restrict__ slot_mapping,
    const uint8_t* __restrict__ compress_lut,
    const int64_t key_tok_stride,
    const int64_t key_head_stride,
    const int64_t value_tok_stride,
    const int64_t value_head_stride,
    const int64_t cache_kv_stride,
    const int64_t cache_block_stride,
    const int64_t cache_slot_stride,
    const int64_t cache_head_stride,
    const int64_t block_size,
    const int64_t num_heads,
    const int64_t num_chunks,
    const int64_t heads_per_block) {
  const int64_t token_idx = blockIdx.x;
  const int64_t z = blockIdx.z;
  const int64_t kv_idx = z / num_chunks;
  const int64_t chunk_idx = z % num_chunks;
  const int thread = threadIdx.x;
  const int local_head = thread / kKVFloat13SignBytes;
  const int pack_thread = thread % kKVFloat13SignBytes;
  const int64_t head_idx = blockIdx.y * heads_per_block + local_head;

  if (head_idx >= num_heads) {
    return;
  }

  const int64_t slot_idx = static_cast<int64_t>(slot_mapping[token_idx]);
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const uint16_t* src = (kv_idx == 0 ? key : value) +
                        token_idx * (kv_idx == 0 ? key_tok_stride : value_tok_stride) +
                        head_idx * (kv_idx == 0 ? key_head_stride : value_head_stride) +
                        chunk_idx * kKVFloat13ChunkSize;
  uint8_t* dst = kv_cache + kv_idx * cache_kv_stride +
                 block_idx * cache_block_stride +
                 block_offset * cache_slot_stride +
                 head_idx * cache_head_stride +
                 chunk_idx * kKVFloat13ChunkBytes;

  uint8_t sign_bits = 0;
  uint32_t exp_hi_packed = 0;
  uint64_t em_packed = 0;

  const int64_t base = static_cast<int64_t>(pack_thread) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint16_t bits = src[base + i];
    const uint8_t sign = static_cast<uint8_t>(bits >> 15);
    const uint8_t exp8 = static_cast<uint8_t>((bits >> 7) & 0xFF);
    const uint8_t mant7 = static_cast<uint8_t>(bits & 0x7F);
    const uint8_t exp5 = compress_lut[exp8];
    const uint8_t exp_hi4 = static_cast<uint8_t>(exp5 >> 1);

    sign_bits |= static_cast<uint8_t>(sign << i);
    if ((i & 1) == 0) {
      exp_hi_packed |= static_cast<uint32_t>(exp_hi4) << ((i >> 1) * 8);
    } else {
      exp_hi_packed |= static_cast<uint32_t>(exp_hi4) << (((i >> 1) * 8) + 4);
    }
    em_packed |=
        static_cast<uint64_t>(((exp5 & 1) << 7) | mant7) << (static_cast<uint64_t>(i) * 8);
  }

  dst[pack_thread] = sign_bits;
  uint8_t* exp_dst =
      dst + kKVFloat13SignBytes + static_cast<int64_t>(pack_thread) * 4;
  uint8_t* em_dst =
      dst + kKVFloat13SignBytes + kKVFloat13ExpHiBytes +
      static_cast<int64_t>(pack_thread) * 8;
  *reinterpret_cast<uint32_t*>(exp_dst) = exp_hi_packed;
  *reinterpret_cast<uint64_t*>(em_dst) = em_packed;
}

#define CALL_RESHAPE_AND_CACHE_KVFLOAT13(INDEX_T)                               \
  reshape_and_cache_kvfloat13_bf16_kernel<INDEX_T>                              \
      <<<grid, block, 0, stream>>>(                                             \
          reinterpret_cast<const uint16_t*>(key.data_ptr()),                    \
          reinterpret_cast<const uint16_t*>(value.data_ptr()),                  \
          reinterpret_cast<uint8_t*>(kv_cache.data_ptr()),                      \
          slot_mapping.data_ptr<INDEX_T>(),                                     \
          compress_lut.data_ptr<uint8_t>(),                                     \
          key.stride(0),                                                        \
          key.stride(1),                                                        \
          value.stride(0),                                                      \
          value.stride(1),                                                      \
          kv_cache.stride(0),                                                   \
          kv_cache.stride(1),                                                   \
          kv_cache.stride(2),                                                   \
          kv_cache.stride(3),                                                   \
          kv_cache.size(2),                                                     \
          key.size(1),                                                          \
          num_chunks,                                                            \
          heads_per_block)

void reshape_and_cache_kvfloat13(torch::Tensor key,
                                 torch::Tensor value,
                                 torch::Tensor kv_cache,
                                 torch::Tensor slot_mapping,
                                 torch::Tensor compress_lut) {
  TORCH_CHECK(key.is_cuda(), "key must be on CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be on CUDA");
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on CUDA");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be on CUDA");
  TORCH_CHECK(compress_lut.is_cuda(), "compress_lut must be on CUDA");

  TORCH_CHECK(key.device() == value.device() && key.device() == kv_cache.device() &&
                  key.device() == slot_mapping.device() &&
                  key.device() == compress_lut.device(),
              "All tensors must be on the same device");

  TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16,
              "key must be bfloat16");
  TORCH_CHECK(value.scalar_type() == at::ScalarType::BFloat16,
              "value must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Byte,
              "kv_cache must be uint8");
  TORCH_CHECK(compress_lut.scalar_type() == at::ScalarType::Byte,
              "compress_lut must be uint8");
  TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int ||
                  slot_mapping.scalar_type() == at::ScalarType::Long,
              "slot_mapping must be int32 or int64");

  TORCH_CHECK(key.dim() == 3 && value.dim() == 3,
              "key and value must be [num_tokens, num_heads, head_size]");
  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must be [2, num_blocks, block_size, num_heads, packed]");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(slot_mapping.size(0) == key.size(0),
              "slot_mapping length must match key/value tokens");
  TORCH_CHECK(compress_lut.numel() == 256, "compress_lut must have 256 entries");
  TORCH_CHECK(key.stride(2) == 1, "key last dimension must be contiguous");
  TORCH_CHECK(value.stride(2) == 1, "value last dimension must be contiguous");
  TORCH_CHECK(kv_cache.stride(4) == 1, "kv_cache packed dimension must be contiguous");

  const int64_t num_tokens = key.size(0);
  if (num_tokens == 0) {
    return;
  }

  const int64_t num_heads = key.size(1);
  const int64_t head_size = key.size(2);
  TORCH_CHECK(head_size % kKVFloat13ChunkSize == 0,
              "head_size must be a multiple of 128");
  const int64_t num_chunks = head_size / kKVFloat13ChunkSize;
  TORCH_CHECK(kv_cache.size(3) == num_heads,
              "kv_cache num_heads must match key/value");
  TORCH_CHECK(kv_cache.size(4) == num_chunks * kKVFloat13ChunkBytes,
              "kv_cache packed size does not match head_size");

  const int64_t heads_per_block = std::min<int64_t>(num_heads, 16);
  const dim3 grid(num_tokens, (num_heads + heads_per_block - 1) / heads_per_block,
                  2 * num_chunks);
  const dim3 block(heads_per_block * kKVFloat13SignBytes);
  const at::cuda::OptionalCUDAGuard device_guard(key.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (slot_mapping.scalar_type() == at::ScalarType::Int) {
    CALL_RESHAPE_AND_CACHE_KVFLOAT13(int32_t);
  } else {
    CALL_RESHAPE_AND_CACHE_KVFLOAT13(int64_t);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def(
      "reshape_and_cache_kvfloat13("
      "Tensor key, Tensor value, Tensor! kv_cache, Tensor slot_mapping, "
      "Tensor compress_lut) -> ()");
}

TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl("reshape_and_cache_kvfloat13", &reshape_and_cache_kvfloat13);
}
