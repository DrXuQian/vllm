#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cub/cub.cuh>

#include <cstdint>
#include <tuple>

namespace {

template <typename seq_t>
__global__ void kvfloat13_row_major_counts_kernel(
    const seq_t* __restrict__ seq_lens,
    int64_t* __restrict__ counts,
    const int64_t batch_size,
    const int64_t block_size,
    const int64_t max_blocks) {
  const int64_t seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (seq_idx >= batch_size) {
    return;
  }

  const int64_t seq_len = static_cast<int64_t>(seq_lens[seq_idx]);
  int64_t num_blocks = 0;
  if (seq_len > 0) {
    num_blocks = (seq_len + block_size - 1) / block_size;
    if (num_blocks > max_blocks) {
      num_blocks = max_blocks;
    }
  }
  counts[seq_idx] = num_blocks;
}

template <typename index_t>
__global__ void kvfloat13_row_major_scatter_kernel(
    const index_t* __restrict__ block_table,
    const int64_t block_table_stride0,
    const int64_t block_table_stride1,
    const int64_t* __restrict__ seq_offsets,
    const int64_t* __restrict__ counts,
    const int64_t* __restrict__ seq_lens,
    index_t* __restrict__ compact_block_table,
    const int64_t compact_block_table_stride0,
    const int64_t compact_block_table_stride1,
    index_t* __restrict__ used_block_ids,
    int64_t* __restrict__ compact_slots,
    const int64_t block_size,
    const bool decode_only) {
  const int64_t seq_idx = blockIdx.x;
  const int64_t num_blocks = counts[seq_idx];
  const int64_t seq_offset = seq_offsets[seq_idx];

  for (int64_t local_block_idx = threadIdx.x; local_block_idx < num_blocks;
       local_block_idx += blockDim.x) {
    const index_t block_id =
        block_table[seq_idx * block_table_stride0 +
                    local_block_idx * block_table_stride1];
    const int64_t compact_row = seq_offset + local_block_idx;
    used_block_ids[compact_row] = block_id;
    compact_block_table[seq_idx * compact_block_table_stride0 +
                        local_block_idx * compact_block_table_stride1] =
        static_cast<index_t>(compact_row);
  }

  if (decode_only && threadIdx.x == 0) {
    const int64_t seq_len = seq_lens[seq_idx];
    if (num_blocks > 0 && seq_len > 0) {
      const int64_t live_pos = seq_len - 1;
      const int64_t local_block_idx = live_pos / block_size;
      compact_slots[seq_idx] =
          (seq_offset + local_block_idx) * block_size + (live_pos % block_size);
    } else {
      compact_slots[seq_idx] = -1;
    }
  }
}

template <typename index_t, typename seq_t>
std::tuple<torch::Tensor, torch::Tensor> build_kvfloat13_row_major_layout_cuda(
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    bool decode_only,
    torch::Tensor compact_block_table) {
  const int64_t batch_size = block_table.size(0);
  const int64_t max_blocks = block_table.size(1);
  auto slots = torch::empty({0}, seq_lens.options().dtype(torch::kLong));

  if (block_table.numel() == 0 || batch_size == 0 || max_blocks == 0) {
    return {
        block_table.new_empty({0}),
        slots,
    };
  }

  auto counts = torch::zeros({batch_size + 1}, seq_lens.options().dtype(torch::kLong));
  auto seq_offsets = torch::empty_like(counts);

  const at::cuda::OptionalCUDAGuard device_guard(block_table.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kCountThreads = 256;
  const int64_t count_blocks = (batch_size + kCountThreads - 1) / kCountThreads;
  kvfloat13_row_major_counts_kernel<seq_t>
      <<<count_blocks, kCountThreads, 0, stream>>>(
          seq_lens.data_ptr<seq_t>(),
          counts.data_ptr<int64_t>(),
          batch_size,
          block_size,
          max_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      temp_storage_bytes,
      counts.data_ptr<int64_t>(),
      seq_offsets.data_ptr<int64_t>(),
      batch_size + 1,
      stream);
  auto temp_storage = torch::empty(
      {static_cast<long>(temp_storage_bytes)},
      block_table.options().dtype(torch::kUInt8));
  cub::DeviceScan::ExclusiveSum(
      temp_storage.data_ptr<uint8_t>(),
      temp_storage_bytes,
      counts.data_ptr<int64_t>(),
      seq_offsets.data_ptr<int64_t>(),
      batch_size + 1,
      stream);
  C10_CUDA_CHECK(cudaGetLastError());

  auto seq_lens_long = seq_lens.to(torch::kLong);

  const int64_t total_used_blocks =
      seq_offsets.index({batch_size}).cpu().item<int64_t>();
  auto used_block_ids = block_table.new_empty({total_used_blocks});

  if (decode_only) {
    slots = torch::empty({batch_size}, seq_lens.options().dtype(torch::kLong));
  }

  if (total_used_blocks == 0) {
    if (decode_only && batch_size > 0) {
      slots.fill_(-1);
    }
    return {
        used_block_ids,
        slots,
    };
  }

  constexpr int kScatterThreads = 128;
  kvfloat13_row_major_scatter_kernel<index_t>
      <<<batch_size, kScatterThreads, 0, stream>>>(
          block_table.data_ptr<index_t>(),
          block_table.stride(0),
          block_table.stride(1),
          seq_offsets.data_ptr<int64_t>(),
          counts.data_ptr<int64_t>(),
          seq_lens_long.data_ptr<int64_t>(),
          compact_block_table.data_ptr<index_t>(),
          compact_block_table.stride(0),
          compact_block_table.stride(1),
          used_block_ids.data_ptr<index_t>(),
          decode_only ? slots.data_ptr<int64_t>() : nullptr,
          block_size,
          decode_only);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {
      used_block_ids,
      slots,
  };
}

std::tuple<torch::Tensor, torch::Tensor> build_kvfloat13_row_major_layout(
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    int64_t block_size,
    bool decode_only,
    torch::Tensor compact_block_table) {
  TORCH_CHECK(block_table.is_cuda(), "block_table must be on CUDA");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be on CUDA");
  TORCH_CHECK(compact_block_table.is_cuda(), "compact_block_table must be on CUDA");
  TORCH_CHECK(
      block_table.device() == seq_lens.device() &&
          block_table.device() == compact_block_table.device(),
      "All tensors must be on the same CUDA device");
  TORCH_CHECK(block_table.dim() == 2, "block_table must be 2D");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be 1D");
  TORCH_CHECK(
      compact_block_table.sizes() == block_table.sizes(),
      "compact_block_table must match block_table shape");
  TORCH_CHECK(
      block_table.size(0) == seq_lens.size(0),
      "block_table batch dimension must match seq_lens");
  TORCH_CHECK(
      compact_block_table.scalar_type() == block_table.scalar_type(),
      "compact_block_table dtype must match block_table dtype");
  TORCH_CHECK(
      block_table.scalar_type() == at::ScalarType::Int ||
          block_table.scalar_type() == at::ScalarType::Long,
      "block_table must be int32 or int64");
  TORCH_CHECK(
      seq_lens.scalar_type() == at::ScalarType::Int ||
          seq_lens.scalar_type() == at::ScalarType::Long,
      "seq_lens must be int32 or int64");

  compact_block_table.zero_();

  if (block_table.scalar_type() == at::ScalarType::Int) {
    if (seq_lens.scalar_type() == at::ScalarType::Int) {
      return build_kvfloat13_row_major_layout_cuda<int32_t, int32_t>(
          block_table,
          seq_lens,
          block_size,
          decode_only,
          compact_block_table);
    }
    return build_kvfloat13_row_major_layout_cuda<int32_t, int64_t>(
        block_table,
        seq_lens,
        block_size,
        decode_only,
        compact_block_table);
  }

  if (seq_lens.scalar_type() == at::ScalarType::Int) {
    return build_kvfloat13_row_major_layout_cuda<int64_t, int32_t>(
        block_table,
        seq_lens,
        block_size,
        decode_only,
        compact_block_table);
  }
  return build_kvfloat13_row_major_layout_cuda<int64_t, int64_t>(
      block_table,
      seq_lens,
      block_size,
      decode_only,
      compact_block_table);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_cache_ops, m) {
  m.def(
      "build_kvfloat13_row_major_layout("
      "Tensor block_table, Tensor seq_lens, int block_size, bool decode_only, "
      "Tensor(a!) compact_block_table) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, m) {
  m.impl(
      "build_kvfloat13_row_major_layout",
      &build_kvfloat13_row_major_layout);
}
