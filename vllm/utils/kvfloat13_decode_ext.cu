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
__global__ void decode_kvfloat13_blocks_bf16_kernel(
    const uint8_t* __restrict__ kv_cache,
    const index_t* __restrict__ used_block_ids,
    uint16_t* __restrict__ out,
    const int64_t cache_kv_stride,
    const int64_t cache_block_stride,
    const int64_t cache_slot_stride,
    const int64_t cache_head_stride,
    const int64_t out_row_stride,
    const int64_t num_used_blocks,
    const int64_t block_size,
    const int64_t num_heads) {
  const int64_t row_idx = blockIdx.x;
  const int64_t chunk_idx = blockIdx.y;
  const int64_t dim_idx = threadIdx.x;

  if (dim_idx >= kKVFloat13ChunkSize) {
    return;
  }

  const int64_t rows_per_kv = num_used_blocks * block_size * num_heads;
  const int64_t kv_idx = row_idx / rows_per_kv;
  const int64_t rem0 = row_idx % rows_per_kv;
  const int64_t local_block_idx = rem0 / (block_size * num_heads);
  const int64_t rem1 = rem0 % (block_size * num_heads);
  const int64_t slot_idx = rem1 / num_heads;
  const int64_t head_idx = rem1 % num_heads;

  const int64_t physical_block_idx =
      static_cast<int64_t>(used_block_ids[local_block_idx]);
  const uint8_t* packed_base = kv_cache + kv_idx * cache_kv_stride +
                               physical_block_idx * cache_block_stride +
                               slot_idx * cache_slot_stride +
                               head_idx * cache_head_stride +
                               chunk_idx * kKVFloat13ChunkBytes;

  const uint8_t sign_byte = packed_base[dim_idx >> 3];
  const uint16_t sign = static_cast<uint16_t>((sign_byte >> (dim_idx & 7)) & 1);

  const uint8_t exp_hi_byte = packed_base[kKVFloat13SignBytes + (dim_idx >> 1)];
  const uint16_t exp_hi4 = static_cast<uint16_t>(
      (dim_idx & 1) == 0 ? (exp_hi_byte & 0x0F) : ((exp_hi_byte >> 4) & 0x0F));

  const uint8_t em =
      packed_base[kKVFloat13SignBytes + kKVFloat13ExpHiBytes + dim_idx];
  const uint16_t exp5 = static_cast<uint16_t>((exp_hi4 << 1) | (em >> 7));
  const uint16_t exp8 = static_cast<uint16_t>(exp5 == 0 ? 0 : exp5 + 100);
  const uint16_t bf16_bits =
      static_cast<uint16_t>((sign << 15) | (exp8 << 7) | (em & 0x7F));

  out[row_idx * out_row_stride + chunk_idx * kKVFloat13ChunkSize + dim_idx] =
      bf16_bits;
}

#define CALL_DECODE_KVFLOAT13_BLOCKS(INDEX_T)                                  \
  decode_kvfloat13_blocks_bf16_kernel<INDEX_T>                                 \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),               \
          used_block_ids.data_ptr<INDEX_T>(),                                  \
          reinterpret_cast<uint16_t*>(out.data_ptr()),                         \
          kv_cache.stride(0),                                                  \
          kv_cache.stride(1),                                                  \
          kv_cache.stride(2),                                                  \
          kv_cache.stride(3),                                                  \
          out_row_stride,                                                      \
          num_used_blocks,                                                     \
          block_size,                                                          \
          num_heads)

void decode_kvfloat13_blocks(torch::Tensor kv_cache,
                             torch::Tensor used_block_ids,
                             torch::Tensor out) {
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on CUDA");
  TORCH_CHECK(used_block_ids.is_cuda(), "used_block_ids must be on CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
  TORCH_CHECK(kv_cache.device() == used_block_ids.device() &&
                  kv_cache.device() == out.device(),
              "All tensors must be on the same device");

  TORCH_CHECK(kv_cache.scalar_type() == at::ScalarType::Byte,
              "kv_cache must be uint8");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16,
              "out must be bfloat16");
  TORCH_CHECK(used_block_ids.scalar_type() == at::ScalarType::Int ||
                  used_block_ids.scalar_type() == at::ScalarType::Long,
              "used_block_ids must be int32 or int64");

  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must be [2, num_blocks, block_size, num_heads, packed]");
  TORCH_CHECK(out.dim() == 5,
              "out must be [2, num_used_blocks, block_size, num_heads, head_size]");

  const int64_t num_used_blocks = used_block_ids.size(0);
  const int64_t block_size = kv_cache.size(2);
  const int64_t num_heads = kv_cache.size(3);
  const int64_t packed_head = kv_cache.size(4);
  const int64_t head_size = out.size(4);
  const int64_t num_chunks = head_size / kKVFloat13ChunkSize;

  TORCH_CHECK(head_size % kKVFloat13ChunkSize == 0,
              "head_size must be a multiple of 128");
  TORCH_CHECK(packed_head == num_chunks * kKVFloat13ChunkBytes,
              "packed head size does not match head_size");
  TORCH_CHECK(out.size(0) == 2 && kv_cache.size(0) == 2, "KV dimension must be 2");
  TORCH_CHECK(out.size(1) == num_used_blocks,
              "out num_used_blocks must match used_block_ids");
  TORCH_CHECK(out.size(2) == block_size, "out block_size mismatch");
  TORCH_CHECK(out.size(3) == num_heads, "out num_heads mismatch");
  TORCH_CHECK(out.stride(4) == 1, "out last dimension must be contiguous");

  if (num_used_blocks == 0) {
    return;
  }

  const int64_t out_row_stride = out.stride(3);
  const dim3 grid(2 * num_used_blocks * block_size * num_heads, num_chunks);
  const dim3 block(kKVFloat13ChunkSize);
  const at::cuda::OptionalCUDAGuard device_guard(kv_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (used_block_ids.scalar_type() == at::ScalarType::Int) {
    CALL_DECODE_KVFLOAT13_BLOCKS(int32_t);
  } else {
    CALL_DECODE_KVFLOAT13_BLOCKS(int64_t);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def(
      "decode_kvfloat13_blocks("
      "Tensor kv_cache, Tensor used_block_ids, Tensor! out) -> ()");
}

TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl("decode_kvfloat13_blocks", &decode_kvfloat13_blocks);
}
