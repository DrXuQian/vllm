#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>

namespace {

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

  compact_block_table.zero_();
  auto slots = torch::empty({0}, seq_lens.options().dtype(torch::kLong));

  if (block_table.numel() == 0) {
    return {
        block_table.new_empty({0}),
        slots,
    };
  }

  const auto max_blocks = block_table.size(1);
  auto num_blocks_per_seq =
      torch::div(seq_lens + (block_size - 1), block_size, "floor");
  auto block_positions = torch::arange(max_blocks, seq_lens.options());
  auto valid_mask =
      block_positions.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1);
  auto used_block_ids = block_table.masked_select(valid_mask);

  if (used_block_ids.numel() > 0) {
    auto compact_ids =
        torch::arange(used_block_ids.numel(), block_table.options());
    compact_block_table.masked_scatter_(valid_mask, compact_ids);
  }

  if (decode_only) {
    auto seq_lens_long = seq_lens.to(torch::kLong);
    auto req_indices = torch::arange(seq_lens.size(0), seq_lens_long.options());
    auto live_pos = seq_lens_long - 1;
    auto local_block_idx = torch::div(live_pos, block_size, "floor");
    auto compact_rows =
        compact_block_table.index({req_indices, local_block_idx}).to(torch::kLong);
    slots = compact_rows * block_size + torch::remainder(live_pos, block_size);
  }

  return {
      used_block_ids,
      slots,
  };
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
