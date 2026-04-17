"""
JIT loader for the fused KVFloat13 decode attention CUDA kernel.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

_module = None

CUDA_SRC_PATH = os.path.join(
    os.path.dirname(__file__), "kvfloat13_fused_decode_attn.cu"
)

WRAPPER_SRC = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

// Forward declaration of the kernel launcher
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
);

torch::Tensor kvfloat13_fused_decode_attn(
    torch::Tensor q,            // [batch, num_qo_heads, head_dim]
    torch::Tensor kv_cache_k,   // [num_pages, page_size, num_kv_heads, packed_bytes]
    torch::Tensor kv_cache_v,   // same
    torch::Tensor page_table,   // [batch, max_num_pages]
    torch::Tensor seq_lens,     // [batch]
    float sm_scale,
    int64_t max_seq_len        // actual max sequence length
) {
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be BF16");

    uint32_t batch_size = q.size(0);
    uint32_t num_qo_heads = q.size(1);
    uint32_t head_dim = q.size(2);
    uint32_t num_kv_heads = kv_cache_k.size(2);
    uint32_t page_size = kv_cache_k.size(1);
    uint32_t max_num_pages_actual = page_table.size(1);  // actual page_table columns

    auto output = torch::empty_like(q);

    kvfloat13_fused_decode_attention(
        q.data_ptr(),
        kv_cache_k.data_ptr(),
        kv_cache_v.data_ptr(),
        output.data_ptr(),
        nullptr,
        page_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_num_pages_actual,  // correct page_table stride
        sm_scale,
        c10::cuda::getCurrentCUDAStream()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kvfloat13_fused_decode_attn", &kvfloat13_fused_decode_attn,
          "KVFloat13 fused decode attention",
          py::arg("q"), py::arg("kv_cache_k"), py::arg("kv_cache_v"),
          py::arg("page_table"), py::arg("seq_lens"),
          py::arg("sm_scale"), py::arg("max_seq_len"));
}
"""


def ensure_kvfloat13_fused_attn_op():
    """JIT compile and load the fused attention kernel."""
    global _module
    if _module is not None:
        return _module

    with open(CUDA_SRC_PATH) as f:
        cuda_src = f.read()

    _module = load_inline(
        name="kvfloat13_fused_attn",
        cpp_sources=[WRAPPER_SRC],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--expt-relaxed-constexpr",
            "-std=c++17",
        ],
        verbose=False,
    )
    return _module


def kvfloat13_fused_decode_attn(
    q: torch.Tensor,
    kv_cache_k: torch.Tensor,
    kv_cache_v: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """
    Fused KVFloat13 decode attention.

    Args:
        q: [batch, num_qo_heads, head_dim] BF16
        kv_cache_k: [num_pages, page_size, num_kv_heads, 208] uint8
        kv_cache_v: [num_pages, page_size, num_kv_heads, 208] uint8
        page_table: [batch, max_num_pages] int32
        seq_lens: [batch] int32
        sm_scale: softmax scale (1/sqrt(head_dim))
        max_seq_len: maximum sequence length (controls grid size)

    Returns:
        output: [batch, num_qo_heads, head_dim] BF16
    """
    if max_seq_len <= 0:
        max_seq_len = int(seq_lens.max().item())
    mod = ensure_kvfloat13_fused_attn_op()
    return mod.kvfloat13_fused_decode_attn(
        q, kv_cache_k, kv_cache_v, page_table, seq_lens, sm_scale, max_seq_len
    )
