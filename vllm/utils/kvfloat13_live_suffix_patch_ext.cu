#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace {

template <typename index_t>
__global__ void kvfloat13_live_suffix_patch_bf16_kernel(
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const index_t* __restrict__ slots,
    const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value,
    const int64_t key_cache_tok_stride,
    const int64_t key_cache_head_stride,
    const int64_t value_cache_tok_stride,
    const int64_t value_cache_head_stride,
    const int64_t key_tok_stride,
    const int64_t key_head_stride,
    const int64_t value_tok_stride,
    const int64_t value_head_stride,
    const int64_t num_heads,
    const int64_t head_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t head_idx = blockIdx.y;
  const int64_t slot_idx = static_cast<int64_t>(slots[token_idx]);
  if (head_idx >= num_heads) {
    return;
  }

  const int64_t key_dst_base =
      slot_idx * key_cache_tok_stride + head_idx * key_cache_head_stride;
  const int64_t value_dst_base =
      slot_idx * value_cache_tok_stride + head_idx * value_cache_head_stride;
  const int64_t key_src_base =
      token_idx * key_tok_stride + head_idx * key_head_stride;
  const int64_t value_src_base =
      token_idx * value_tok_stride + head_idx * value_head_stride;

  for (int64_t dim_idx = threadIdx.x; dim_idx < head_size; dim_idx += blockDim.x) {
    key_cache[key_dst_base + dim_idx] = key[key_src_base + dim_idx];
    value_cache[value_dst_base + dim_idx] = value[value_src_base + dim_idx];
  }
}

#define CALL_KVFLOAT13_LIVE_SUFFIX_PATCH(INDEX_T)                           \
  kvfloat13_live_suffix_patch_bf16_kernel<INDEX_T>                          \
      <<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<uint16_t*>(key_cache.data_ptr()),                \
          reinterpret_cast<uint16_t*>(value_cache.data_ptr()),              \
          slots.data_ptr<INDEX_T>(),                                        \
          reinterpret_cast<const uint16_t*>(key.data_ptr()),                \
          reinterpret_cast<const uint16_t*>(value.data_ptr()),              \
          key_cache.stride(0),                                              \
          key_cache.stride(1),                                              \
          value_cache.stride(0),                                            \
          value_cache.stride(1),                                            \
          key.stride(0),                                                    \
          key.stride(1),                                                    \
          value.stride(0),                                                  \
          value.stride(1),                                                  \
          key.size(1),                                                      \
          key.size(2))

void kvfloat13_live_suffix_patch(torch::Tensor key_cache,
                                 torch::Tensor value_cache,
                                 torch::Tensor slots,
                                 torch::Tensor key,
                                 torch::Tensor value) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be on CUDA");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be on CUDA");
  TORCH_CHECK(slots.is_cuda(), "slots must be on CUDA");
  TORCH_CHECK(key.is_cuda(), "key must be on CUDA");
  TORCH_CHECK(value.is_cuda(), "value must be on CUDA");

  TORCH_CHECK(key_cache.device() == value_cache.device() &&
                  key_cache.device() == slots.device() &&
                  key_cache.device() == key.device() &&
                  key_cache.device() == value.device(),
              "All tensors must be on the same device");

  TORCH_CHECK(key_cache.scalar_type() == at::ScalarType::BFloat16,
              "key_cache must be bfloat16");
  TORCH_CHECK(value_cache.scalar_type() == at::ScalarType::BFloat16,
              "value_cache must be bfloat16");
  TORCH_CHECK(key.scalar_type() == at::ScalarType::BFloat16,
              "key must be bfloat16");
  TORCH_CHECK(value.scalar_type() == at::ScalarType::BFloat16,
              "value must be bfloat16");
  TORCH_CHECK(slots.scalar_type() == at::ScalarType::Int ||
                  slots.scalar_type() == at::ScalarType::Long,
              "slots must be int32 or int64");

  TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
              "key_cache and value_cache must be 3D");
  TORCH_CHECK(key.dim() == 3 && value.dim() == 3, "key and value must be 3D");
  TORCH_CHECK(slots.is_contiguous(), "slots must be contiguous");
  TORCH_CHECK(key_cache.stride(2) == 1, "key_cache last dim must be contiguous");
  TORCH_CHECK(value_cache.stride(2) == 1,
              "value_cache last dim must be contiguous");
  TORCH_CHECK(key.stride(2) == 1, "key last dim must be contiguous");
  TORCH_CHECK(value.stride(2) == 1, "value last dim must be contiguous");

  TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
              "key_cache and value_cache must have the same shape");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value must have the same shape");
  TORCH_CHECK(key_cache.size(1) == key.size(1) &&
                  key_cache.size(2) == key.size(2),
              "Cache and live K/V head dimensions must match");
  TORCH_CHECK(key.size(0) == slots.size(0),
              "key/value token dimension must match slots");

  const int64_t num_tokens = slots.size(0);
  if (num_tokens == 0) {
    return;
  }

  const int block = (key.size(2) > 128) ? 256 : 128;
  const dim3 grid(num_tokens, key.size(1));
  const at::cuda::OptionalCUDAGuard device_guard(key_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (slots.scalar_type() == at::ScalarType::Int) {
    CALL_KVFLOAT13_LIVE_SUFFIX_PATCH(int32_t);
  } else {
    CALL_KVFLOAT13_LIVE_SUFFIX_PATCH(int64_t);
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def(
      "kvfloat13_live_suffix_patch("
      "Tensor! key_cache, Tensor! value_cache, Tensor slots, "
      "Tensor key, Tensor value) -> ()");
}

TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl("kvfloat13_live_suffix_patch", &kvfloat13_live_suffix_patch);
}
