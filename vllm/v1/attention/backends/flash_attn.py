# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import copy
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_supports_fp8,
    flash_attn_supports_quant_query_input,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.worker.workspace import current_workspace_manager

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_supports_sinks,
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
import vllm.envs as envs
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    get_layers_from_vllm_config,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.kvfloat13 import (
    build_kvfloat13_row_major_layout,
    decode_kvfloat13,
    decode_kvfloat13_blocks_triton,
    is_kvfloat13_kv_cache,
    kvfloat13_packed_bytes_per_head,
    reshape_and_cache_kvfloat13,
)
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)
_KVFLOAT13_NVTX_PROFILE = os.getenv("VLLM_KVFLOAT13_NVTX_PROFILE", "0") == "1"


@contextmanager
def _nvtx_range(name: str):
    if _KVFLOAT13_NVTX_PROFILE:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "kfloat13",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        vllm_config = get_current_vllm_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        if (
            model_config
            and model_config.is_hybrid
            and (
                cache_config.mamba_ssm_cache_dtype == "float32"
                or cache_config.mamba_cache_dtype == "float32"
            )
        ):
            # NOTE(tdoublep): while in principle, FA supports
            # MultipleOf(16), these are the block sizes that do not
            # suffer from the NaN propagation problem described here:
            # https://github.com/Dao-AILab/flash-attention/issues/1974
            return [16, 32, 64]
        return [MultipleOf(16)]

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        if is_kvfloat13_kv_cache(cache_dtype_str):
            return (
                2,
                num_blocks,
                block_size,
                num_kv_heads,
                kvfloat13_packed_bytes_per_head(head_size),
            )
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (2, 0, 1, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, 2, block_size, head_size)
            return (2, 4, 0, 1, 3, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if is_kvfloat13_kv_cache(kv_cache_dtype):
            return True
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_fp8()
        return kv_cache_dtype in ["auto", "float16", "bfloat16"]

    @classmethod
    def supports_sink(cls) -> bool:
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        return None


@dataclass
class FlashAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True


def _get_sliding_window_configs(
    vllm_config: VllmConfig,
) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, FlashAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from vllm.distributed.parallel_state import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        self.cp_kv_cache_interleave_size = (
            self.parallel_config.cp_kv_cache_interleave_size
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        if self.dcp_world_size > 1:
            max_num_reqs = vllm_config.scheduler_config.max_num_seqs
            self._dcp_context_kv_lens = torch.zeros(
                max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                    cache_dtype
                )
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=max_num_splits,
                )
            return None

        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        if self.dcp_world_size > 1:
            query_lens = query_start_loc[1:] - query_start_loc[:-1]
            context_kv_lens = seq_lens - query_lens
            local_context_kv_lens = get_dcp_local_seq_lens(
                context_kv_lens,
                self.dcp_world_size,
                self.dcp_rank,
                self.cp_kv_cache_interleave_size,
            )
            self._dcp_context_kv_lens[:num_reqs] = local_context_kv_lens
            self._dcp_context_kv_lens[num_reqs:] = 0
            dcp_context_kv_lens = self._dcp_context_kv_lens[:num_reqs]

            # After DCP distribution, the maximum number of tokens for any rank is
            # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
            # and I is cp_kv_cache_interleave_size.
            # This eliminates GPU->CPU sync while minimizing workspace over-allocation.
            num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
            max_dcp_context_kv_len = (
                (max_seq_len + num_partitions - 1) // num_partitions
            ) * self.cp_kv_cache_interleave_size

            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=dcp_context_kv_lens,
                max_seq_len=max_dcp_context_kv_len,
                causal=False,
            )
        elif use_cascade:
            cu_prefix_query_lens = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device=self.device
            )
            prefix_kv_lens = torch.tensor(
                [common_prefix_len], dtype=torch.int32, device=self.device
            )
            # Use GPU tensor directly - no CPU sync needed
            suffix_kv_lens = seq_lens[:num_reqs] - common_prefix_len
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False,
            )
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=suffix_kv_lens,
                max_seq_len=max_seq_len - common_prefix_len,
                causal=True,
            )
        else:
            scheduler_metadata = schedule(
                batch_size=num_reqs,
                cu_query_lens=query_start_loc,
                max_query_len=max_query_len,
                seqlens=seq_lens,
                max_seq_len=max_seq_len,
                causal=causal,
            )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def update_block_table(
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)


class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
        )
        # head_size > 256 requires FA4 on SM90+; force upgrade from FA3
        if (
            head_size > 256
            and self.vllm_flash_attn_version == 3
            and current_platform.is_cuda()
            and current_platform.is_device_capability_family(90)
        ):
            self.vllm_flash_attn_version = 4
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
            scope="local",
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        vllm_config = get_current_vllm_config_or_none()
        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        self.dcp_combine = dcp_a2a_lse_reduce if dcp_a2a else cp_lse_ag_out_rs

        self._dcp_dtype: torch.dtype | None = None
        if vllm_config is not None and self.dcp_world_size > 1:
            self._dcp_dtype = vllm_config.model_config.dtype
        self._enable_prefix_caching = bool(
            vllm_config is not None and vllm_config.cache_config.enable_prefix_caching
        )
        self._kvfloat13_allow_dynamic_shadow_growth = not (
            vllm_config is not None
            and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self._kvfloat13_use_live_suffix_patch_kernel = (
            envs.VLLM_KVFLOAT13_USE_LIVE_SUFFIX_PATCH_KERNEL
        )
        self._kvfloat13_shadow_kv: torch.Tensor | None = None
        self._kvfloat13_shadow_seq_len = 0
        self._kvfloat13_decode_kv: torch.Tensor | None = None
        self._kvfloat13_single_cu_seqlens_k: torch.Tensor | None = None
        self._kvfloat13_block_positions: torch.Tensor | None = None
        self._kvfloat13_compact_block_table: torch.Tensor | None = None
        self._kvfloat13_batched_shadow_kv: torch.Tensor | None = None
        self._kvfloat13_batched_shadow_generation = -1
        self._kvfloat13_row_major_decode_cache: dict[str, object] | None = None
        self._kvfloat13_row_major_generation = 0
        self._kvfloat13_graph_dense_block_table: torch.Tensor | None = None
        self._kvfloat13_graph_dense_base_slots: torch.Tensor | None = None
        self._kvfloat13_graph_live_pos_buf: torch.Tensor | None = None
        self._kvfloat13_graph_block_idx_buf: torch.Tensor | None = None
        self._kvfloat13_graph_compact_slots: torch.Tensor | None = None
        self._kvfloat13_graph_prev_block_ids: torch.Tensor | None = None

    def _can_use_kvfloat13_single_request_fast_path(
        self,
        query: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
    ) -> bool:
        if not is_kvfloat13_kv_cache(self.kv_cache_dtype):
            return False
        if attn_metadata.use_cascade or self.dcp_world_size != 1:
            return False
        if attn_metadata.block_table.shape[0] != 1:
            return False
        return True

    def _can_use_kvfloat13_batched_graph_flat_path(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> bool:
        if not is_kvfloat13_kv_cache(self.kv_cache_dtype):
            return False
        if self._kvfloat13_allow_dynamic_shadow_growth:
            return False
        if attn_metadata.use_cascade:
            return False
        if self.dcp_world_size != 1:
            return False
        if attn_metadata.block_table.shape[0] <= 1:
            return False
        return (
            attn_metadata.max_query_len == 1
            and attn_metadata.num_actual_tokens == attn_metadata.seq_lens.shape[0]
        )

    def _get_kvfloat13_block_positions(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        max_blocks = block_table.shape[1]
        block_positions = self._kvfloat13_block_positions
        if (
            block_positions is None
            or block_positions.device != block_table.device
            or block_positions.dtype != seq_lens.dtype
            or block_positions.shape[0] < max_blocks
        ):
            block_positions = torch.arange(
                max_blocks,
                device=block_table.device,
                dtype=seq_lens.dtype,
            )
            self._kvfloat13_block_positions = block_positions
        return block_positions[:max_blocks]

    def _get_kvfloat13_compact_block_table(
        self,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        compact_block_table = self._kvfloat13_compact_block_table
        if (
            compact_block_table is None
            or compact_block_table.device != block_table.device
            or compact_block_table.dtype != block_table.dtype
            or compact_block_table.shape[0] < block_table.shape[0]
            or compact_block_table.shape[1] < block_table.shape[1]
        ):
            compact_block_table = torch.empty_like(block_table)
            self._kvfloat13_compact_block_table = compact_block_table
        compact_block_table = compact_block_table[
            : block_table.shape[0], : block_table.shape[1]
        ]
        compact_block_table.zero_()
        return compact_block_table

    def _get_kvfloat13_decode_buffer(
        self,
        kv_cache: torch.Tensor,
        num_used_blocks: int,
    ) -> torch.Tensor:
        decode_kv = self._kvfloat13_decode_kv
        block_size = kv_cache.shape[2]
        rows_needed = 2 * num_used_blocks * block_size * self.num_kv_heads
        if (
            decode_kv is None
            or decode_kv.device != kv_cache.device
            or decode_kv.shape[1] != self.head_size
            or decode_kv.shape[0] < rows_needed
        ):
            current_capacity = (
                decode_kv.shape[0]
                if decode_kv is not None and decode_kv.shape[1] == self.head_size
                else 0
            )
            new_capacity = max(rows_needed, 2 * current_capacity, 1)
            decode_kv = torch.empty(
                (new_capacity, self.head_size),
                device=kv_cache.device,
                dtype=torch.bfloat16,
            )
            self._kvfloat13_decode_kv = decode_kv
        return decode_kv[:rows_needed].view(
            2,
            num_used_blocks,
            block_size,
            self.num_kv_heads,
            self.head_size,
        )

    def _get_kvfloat13_single_cu_seqlens_k(
        self,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        cu_seqlens_k = self._kvfloat13_single_cu_seqlens_k
        if (
            cu_seqlens_k is None
            or cu_seqlens_k.device != query_start_loc.device
            or cu_seqlens_k.dtype != query_start_loc.dtype
        ):
            cu_seqlens_k = torch.empty(
                2,
                device=query_start_loc.device,
                dtype=query_start_loc.dtype,
            )
            self._kvfloat13_single_cu_seqlens_k = cu_seqlens_k
        cu_seqlens_k[:1].copy_(query_start_loc[:1])
        cu_seqlens_k[1:].copy_(seq_lens[:1])
        return cu_seqlens_k

    def _get_kvfloat13_batched_shadow_buffer(
        self,
        kv_cache: torch.Tensor,
        num_used_blocks: int,
    ) -> torch.Tensor:
        shadow_kv = self._kvfloat13_batched_shadow_kv
        block_size = kv_cache.shape[2]
        if (
            shadow_kv is None
            or shadow_kv.device != kv_cache.device
            or shadow_kv.dtype != torch.bfloat16
            or shadow_kv.shape[1] != num_used_blocks
            or shadow_kv.shape[2] != block_size
            or shadow_kv.shape[3] != self.num_kv_heads
            or shadow_kv.shape[4] != self.head_size
        ):
            shadow_kv = torch.empty(
                (2, num_used_blocks, block_size, self.num_kv_heads, self.head_size),
                device=kv_cache.device,
                dtype=torch.bfloat16,
            )
            self._kvfloat13_batched_shadow_kv = shadow_kv
        return shadow_kv

    def _get_kvfloat13_graph_dense_block_table(
        self,
        block_table: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_blocks = block_table.shape
        dense_block_table = self._kvfloat13_graph_dense_block_table
        if (
            dense_block_table is None
            or dense_block_table.device != block_table.device
            or dense_block_table.dtype != block_table.dtype
            or dense_block_table.shape[0] < batch_size
            or dense_block_table.shape[1] < max_blocks
        ):
            dense_block_table = torch.arange(
                batch_size * max_blocks,
                device=block_table.device,
                dtype=block_table.dtype,
            ).view(batch_size, max_blocks)
            self._kvfloat13_graph_dense_block_table = dense_block_table
        dense_block_table = dense_block_table[:batch_size, :max_blocks]

        base_slots = self._kvfloat13_graph_dense_base_slots
        if (
            base_slots is None
            or base_slots.device != block_table.device
            or base_slots.dtype != torch.long
            or base_slots.shape[0] < batch_size
        ):
            base_slots = (
                torch.arange(
                    batch_size,
                    device=block_table.device,
                    dtype=torch.long,
                )
                * (max_blocks * block_size)
            )
            self._kvfloat13_graph_dense_base_slots = base_slots
        return dense_block_table, base_slots[:batch_size]

    def _get_kvfloat13_graph_compact_slots(
        self,
        seq_lens: torch.Tensor,
        base_slots: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch_size = seq_lens.shape[0]
        compact_slots = self._kvfloat13_graph_compact_slots
        live_pos_buf = self._kvfloat13_graph_live_pos_buf
        block_idx_buf = self._kvfloat13_graph_block_idx_buf
        if (
            compact_slots is None
            or compact_slots.device != seq_lens.device
            or compact_slots.dtype != torch.long
            or compact_slots.shape[0] < batch_size
        ):
            compact_slots = torch.empty(
                batch_size,
                device=seq_lens.device,
                dtype=torch.long,
            )
            self._kvfloat13_graph_compact_slots = compact_slots
        if (
            live_pos_buf is None
            or live_pos_buf.device != seq_lens.device
            or live_pos_buf.dtype != torch.long
            or live_pos_buf.shape[0] < batch_size
        ):
            live_pos_buf = torch.empty(
                batch_size,
                device=seq_lens.device,
                dtype=torch.long,
            )
            self._kvfloat13_graph_live_pos_buf = live_pos_buf
        if (
            block_idx_buf is None
            or block_idx_buf.device != seq_lens.device
            or block_idx_buf.dtype != torch.long
            or block_idx_buf.shape[0] < batch_size
        ):
            block_idx_buf = torch.empty(
                batch_size,
                device=seq_lens.device,
                dtype=torch.long,
            )
            self._kvfloat13_graph_block_idx_buf = block_idx_buf

        compact_slots = compact_slots[:batch_size]
        live_pos_buf = live_pos_buf[:batch_size]
        block_idx_buf = block_idx_buf[:batch_size]
        live_pos_buf.copy_(seq_lens)
        live_pos_buf.sub_(1)
        block_idx_buf.copy_(live_pos_buf)
        block_idx_buf.floor_divide_(block_size)
        compact_slots.copy_(base_slots)
        compact_slots.add_(block_idx_buf, alpha=block_size)
        live_pos_buf.remainder_(block_size)
        compact_slots.add_(live_pos_buf)
        return compact_slots

    def _try_get_kvfloat13_batched_shadow_cache(
        self,
        kv_cache: torch.Tensor,
        layout_generation: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        shadow_kv = self._kvfloat13_batched_shadow_kv
        block_size = kv_cache.shape[2]
        if (
            shadow_kv is None
            or shadow_kv.device != kv_cache.device
            or shadow_kv.shape[2] != block_size
            or shadow_kv.shape[3] != self.num_kv_heads
            or shadow_kv.shape[4] != self.head_size
            or self._kvfloat13_batched_shadow_generation != layout_generation
        ):
            return None

        return shadow_kv[0], shadow_kv[1]

    def _get_kvfloat13_batched_shadow_cache(
        self,
        kv_cache: torch.Tensor,
        used_block_ids: torch.Tensor,
        layout_generation: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._try_get_kvfloat13_batched_shadow_cache(
            kv_cache, layout_generation
        )
        if cached is not None:
            with _nvtx_range("fa.kvfloat13.shadow_hit"):
                return cached

        num_used_blocks = int(used_block_ids.numel())
        shadow_kv = self._get_kvfloat13_batched_shadow_buffer(kv_cache, num_used_blocks)
        with _nvtx_range("fa.kvfloat13.decode_to_shadow"):
            decode_kvfloat13_blocks_triton(
                kv_cache,
                used_block_ids,
                self.head_size,
                out=shadow_kv,
            )
        self._kvfloat13_batched_shadow_generation = layout_generation
        return shadow_kv[0], shadow_kv[1]

    def _set_kvfloat13_batched_shadow_cache(
        self,
        kv_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layout_generation: int,
    ) -> None:
        num_used_blocks = key_cache.shape[0]
        shadow_kv = self._get_kvfloat13_batched_shadow_buffer(kv_cache, num_used_blocks)
        shadow_kv[0].copy_(key_cache)
        shadow_kv[1].copy_(value_cache)
        self._kvfloat13_batched_shadow_generation = layout_generation

    def _get_kvfloat13_row_major_cache(
        self,
        attn_metadata: FlashAttentionMetadata,
        block_size: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[int],
        list[int],
        torch.Tensor | None,
        torch.Tensor | None,
        int,
    ]:
        cache = getattr(attn_metadata, "_kvfloat13_row_major_cache", None)
        if cache is not None and cache["block_size"] == block_size:
            return (
                cache["used_block_ids"],
                cache["compact_block_table"],
                cache["seq_lens_list"],
                cache["num_blocks_list"],
                cache["query_lens"],
                cache["compact_slots"],
                cache["prefill_slots"],
                cache["generation"],
            )

        seq_lens = attn_metadata.seq_lens
        block_table = attn_metadata.block_table
        is_decode_only = (
            not self._enable_prefix_caching
            and attn_metadata.max_query_len == 1
            and attn_metadata.num_actual_tokens == seq_lens.shape[0]
        )
        num_blocks_per_seq = torch.div(
            seq_lens + block_size - 1,
            block_size,
            rounding_mode="floor",
        )
        layout_generation = self._kvfloat13_row_major_generation
        cached_layout_hit = False
        compact_slots = None

        cached_decode = self._kvfloat13_row_major_decode_cache
        if (
            is_decode_only
            and cached_decode is not None
            and cached_decode["block_size"] == block_size
            and cached_decode["device"] == block_table.device
            and cached_decode["dtype"] == block_table.dtype
            and cached_decode["shape"] == tuple(block_table.shape)
        ):
            cached_num_blocks = cached_decode["num_blocks_per_seq"]
            if torch.equal(cached_num_blocks, num_blocks_per_seq):
                first_block_ids = block_table[:, 0]
                last_block_positions = (num_blocks_per_seq - 1).clamp_min(0).to(
                    torch.long
                )
                req_indices = torch.arange(
                    block_table.shape[0],
                    device=block_table.device,
                    dtype=torch.long,
                )
                last_block_ids = block_table[req_indices, last_block_positions]
                if torch.equal(cached_decode["first_block_ids"], first_block_ids) and (
                    torch.equal(cached_decode["last_block_ids"], last_block_ids)
                ):
                    used_block_ids = cached_decode["used_block_ids"]
                    compact_block_table = cached_decode["compact_block_table"]
                    compact_slots = cached_decode["compact_slots"]
                    layout_generation = int(cached_decode["generation"])
                    cached_layout_hit = True
                else:
                    cached_decode = None
            else:
                cached_decode = None
        else:
            cached_decode = None

        if not cached_layout_hit:
            with _nvtx_range("fa.kvfloat13.row_major_layout"):
                compact_block_table = self._get_kvfloat13_compact_block_table(block_table)
                used_block_ids, compact_block_table, compact_slots = build_kvfloat13_row_major_layout(
                    block_table,
                    seq_lens,
                    block_size,
                    decode_only=is_decode_only,
                    block_positions=self._get_kvfloat13_block_positions(
                        block_table, seq_lens
                    ),
                    compact_block_table=compact_block_table,
                )
            self._kvfloat13_row_major_generation += 1
            layout_generation = self._kvfloat13_row_major_generation
            if is_decode_only:
                req_indices = torch.arange(
                    block_table.shape[0],
                    device=block_table.device,
                    dtype=torch.long,
                )
                last_block_positions = (num_blocks_per_seq - 1).clamp_min(0).to(
                    torch.long
                )
                self._kvfloat13_row_major_decode_cache = {
                    "block_size": block_size,
                    "device": block_table.device,
                    "dtype": block_table.dtype,
                    "shape": tuple(block_table.shape),
                    "num_blocks_per_seq": num_blocks_per_seq.clone(),
                    "first_block_ids": block_table[:, 0].clone(),
                    "last_block_ids": block_table[req_indices, last_block_positions].clone(),
                    "used_block_ids": used_block_ids,
                    "compact_block_table": compact_block_table,
                    "compact_slots": compact_slots,
                    "generation": layout_generation,
                }
            else:
                self._kvfloat13_row_major_decode_cache = None

        if is_decode_only:
            seq_lens_list: list[int] = []
            num_blocks_list: list[int] = []
            query_lens: list[int] = []
            prefill_slots = None
        else:
            seq_lens_list = [int(x) for x in seq_lens.tolist()]
            num_blocks_list = [cdiv(seq_len, block_size) for seq_len in seq_lens_list]
            query_lens = [
                int(x) for x in torch.diff(attn_metadata.query_start_loc).tolist()
            ]
            prefill_slots = None

        if is_decode_only and compact_slots is None:
            seq_lens_long = seq_lens.to(torch.long)
            req_indices = torch.arange(
                seq_lens.shape[0],
                device=seq_lens.device,
                dtype=torch.long,
            )
            live_pos = seq_lens_long - 1
            local_block_idx = torch.div(
                live_pos,
                block_size,
                rounding_mode="floor",
            )
            compact_rows = compact_block_table[
                req_indices,
                local_block_idx,
            ].to(torch.long)
            compact_slots = compact_rows * block_size + (live_pos % block_size)
        elif (
            not is_decode_only
            and not self._enable_prefix_caching
            and query_lens
            and all(
                query_len == seq_len
                for seq_len, query_len in zip(
                    seq_lens_list, query_lens, strict=True
                )
            )
        ):
            seq_lens_long = seq_lens.to(torch.long)
            seq_indices = torch.repeat_interleave(
                torch.arange(
                    seq_lens.shape[0],
                    device=seq_lens.device,
                    dtype=torch.long,
                ),
                seq_lens_long,
            )
            token_offsets = torch.repeat_interleave(
                attn_metadata.query_start_loc[:-1].to(torch.long),
                seq_lens_long,
            )
            token_pos = torch.arange(
                int(seq_lens_long.sum().item()),
                device=seq_lens.device,
                dtype=torch.long,
            )
            local_pos = token_pos - token_offsets
            compact_rows = compact_block_table[
                seq_indices,
                torch.div(local_pos, block_size, rounding_mode="floor"),
            ].to(torch.long)
            prefill_slots = compact_rows * block_size + (local_pos % block_size)

        cache = {
            "block_size": block_size,
            "used_block_ids": used_block_ids,
            "compact_block_table": compact_block_table,
            "seq_lens_list": seq_lens_list,
            "num_blocks_list": num_blocks_list,
            "query_lens": query_lens,
            "compact_slots": compact_slots,
            "prefill_slots": prefill_slots,
            "generation": layout_generation,
        }
        setattr(attn_metadata, "_kvfloat13_row_major_cache", cache)
        return (
            used_block_ids,
            compact_block_table,
            seq_lens_list,
            num_blocks_list,
            query_lens,
            compact_slots,
            prefill_slots,
            layout_generation,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        block_table = attn_metadata.block_table

        if self._can_use_kvfloat13_single_request_fast_path(query, attn_metadata):
            return self._forward_kvfloat13_single_request(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        if self._can_use_kvfloat13_batched_graph_flat_path(attn_metadata):
            return self._forward_kvfloat13_batched_graph_flat(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        if (
            is_kvfloat13_kv_cache(self.kv_cache_dtype)
            and not attn_metadata.use_cascade
            and self.dcp_world_size == 1
        ):
            return self._forward_kvfloat13_batched_dense(
                layer=layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )

        if is_kvfloat13_kv_cache(self.kv_cache_dtype):
            key_cache, value_cache, block_table = self._decode_kvfloat13_blocks(
                kv_cache,
                block_table,
                attn_metadata.seq_lens,
            )
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                self.kv_cache_dtype
            )
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            q_descale = (
                layer._q_scale.expand(descale_shape)
                if self.supports_quant_query_input
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    block_table=block_table,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                with _nvtx_range("fa.forward.default.flash_attn"):
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                    )
                return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def _forward_kvfloat13_single_request(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_actual_tokens = attn_metadata.num_actual_tokens
        block_size = kv_cache.shape[2]
        seq_len = attn_metadata.max_seq_len
        start_pos = seq_len - num_actual_tokens
        num_used_blocks = cdiv(seq_len, block_size)
        used_block_ids = attn_metadata.block_table[0, :num_used_blocks]
        with _nvtx_range("fa.kvfloat13.single.decode"):
            decoded_kv = self._get_kvfloat13_decode_buffer(
                kv_cache,
                num_used_blocks,
            )
            decode_kvfloat13_blocks_triton(
                kv_cache,
                used_block_ids,
                self.head_size,
                out=decoded_kv,
            )
            decoded_shape = (
                num_used_blocks * block_size,
                self.num_kv_heads,
                self.head_size,
            )
            key_cache = decoded_kv[0].reshape(decoded_shape)[:seq_len]
            value_cache = decoded_kv[1].reshape(decoded_shape)[:seq_len]

        if num_actual_tokens > 0:
            live_key = key[:num_actual_tokens]
            live_value = value[:num_actual_tokens]
            with _nvtx_range("fa.kvfloat13.single.live_suffix_patch"):
                key_cache[start_pos:seq_len].copy_(live_key)
                value_cache[start_pos:seq_len].copy_(live_value)
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = self._get_kvfloat13_single_cu_seqlens_k(
            cu_seqlens_q,
            attn_metadata.seq_lens,
        )
        max_seqlen_q = attn_metadata.max_query_len
        scheduler_metadata = attn_metadata.scheduler_metadata
        descale_shape = (1, self.num_kv_heads)

        q_descale = (
            layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )

        with _nvtx_range("fa.kvfloat13.single.flash_attn"):
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
        return output

    def _forward_kvfloat13_batched_graph_flat(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        block_size = kv_cache.shape[2]
        batch_size, max_blocks = attn_metadata.block_table.shape
        num_dense_blocks = batch_size * max_blocks
        shadow_kv = self._get_kvfloat13_batched_shadow_buffer(
            kv_cache,
            num_dense_blocks,
        )
        dense_block_table, base_slots = self._get_kvfloat13_graph_dense_block_table(
            attn_metadata.block_table,
            block_size,
        )
        compact_slots = self._get_kvfloat13_graph_compact_slots(
            attn_metadata.seq_lens,
            base_slots,
            block_size,
        )
        flat_block_ids = attn_metadata.block_table.reshape(-1)

        with _nvtx_range("fa.kvfloat13.graph_batched.decode"):
            # Note: During CUDA graph capture/replay, we must use the
            # full decode path (no data-dependent branching allowed).
            # The graph_flat path is specifically for cudagraph mode,
            # so always do full decode here.
            decode_kvfloat13_blocks_triton(
                kv_cache,
                flat_block_ids,
                self.head_size,
                out=shadow_kv,
            )

        with _nvtx_range("fa.kvfloat13.graph_batched.live_suffix_patch"):
            flat_key_cache = shadow_kv[0].view(-1, self.num_kv_heads, self.head_size)
            flat_value_cache = shadow_kv[1].view(
                -1,
                self.num_kv_heads,
                self.head_size,
            )
            live_key = key[: attn_metadata.num_actual_tokens]
            live_value = value[: attn_metadata.num_actual_tokens]
            if self._kvfloat13_use_live_suffix_patch_kernel:
                ops.kvfloat13_live_suffix_patch(
                    flat_key_cache,
                    flat_value_cache,
                    compact_slots,
                    live_key,
                    live_value,
                )
            else:
                flat_key_cache.index_copy_(0, compact_slots, live_key)
                flat_value_cache.index_copy_(0, compact_slots, live_value)

        cu_seqlens_q = attn_metadata.query_start_loc
        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
        q_descale = (
            layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        with _nvtx_range("fa.kvfloat13.graph_batched.flash_attn"):
            flash_attn_varlen_func(
                q=query[: attn_metadata.num_actual_tokens],
                k=shadow_kv[0],
                v=shadow_kv[1],
                out=output[: attn_metadata.num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=dense_block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
        return output

    def _forward_kvfloat13_batched_dense(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        block_size = kv_cache.shape[2]
        (
            used_block_ids,
            compact_block_table,
            seq_lens_list,
            num_blocks_list,
            query_lens,
            compact_slots,
            prefill_slots,
            layout_generation,
        ) = self._get_kvfloat13_row_major_cache(
            attn_metadata,
            block_size,
        )
        full_live_prefill = (
            compact_slots is None
            and not self._enable_prefix_caching
            and all(
                query_len == seq_len
                for seq_len, query_len in zip(seq_lens_list, query_lens, strict=True)
            )
        )
        if compact_slots is not None and not self._enable_prefix_caching:
            with _nvtx_range("fa.kvfloat13.batched.shadow_or_decode"):
                key_cache, value_cache = self._get_kvfloat13_batched_shadow_cache(
                    kv_cache,
                    used_block_ids,
                    layout_generation,
                )
        elif full_live_prefill:
            shadow_kv = self._get_kvfloat13_batched_shadow_buffer(
                kv_cache,
                int(used_block_ids.numel()),
            )
            key_cache, value_cache = shadow_kv[0], shadow_kv[1]
        else:
            with _nvtx_range("fa.kvfloat13.batched.decode_pair"):
                key_cache, value_cache = self._decode_kvfloat13_pair(
                    kv_cache,
                    used_block_ids,
                )

        if compact_slots is not None:
            with _nvtx_range("fa.kvfloat13.batched.live_suffix_patch"):
                flat_key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
                flat_value_cache = value_cache.view(
                    -1,
                    self.num_kv_heads,
                    self.head_size,
                )
                live_key = key[: attn_metadata.num_actual_tokens]
                live_value = value[: attn_metadata.num_actual_tokens]
                if self._kvfloat13_use_live_suffix_patch_kernel:
                    ops.kvfloat13_live_suffix_patch(
                        flat_key_cache,
                        flat_value_cache,
                        compact_slots,
                        live_key,
                        live_value,
                    )
                else:
                    flat_key_cache.index_copy_(0, compact_slots, live_key)
                    flat_value_cache.index_copy_(0, compact_slots, live_value)
        else:
            with _nvtx_range("fa.kvfloat13.batched.prefill_copy"):
                if prefill_slots is not None:
                    flat_key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
                    flat_value_cache = value_cache.view(
                        -1,
                        self.num_kv_heads,
                        self.head_size,
                    )
                    live_key = key[: attn_metadata.num_actual_tokens]
                    live_value = value[: attn_metadata.num_actual_tokens]
                    if self._kvfloat13_use_live_suffix_patch_kernel:
                        ops.kvfloat13_live_suffix_patch(
                            flat_key_cache,
                            flat_value_cache,
                            prefill_slots,
                            live_key,
                            live_value,
                        )
                    else:
                        flat_key_cache.index_copy_(0, prefill_slots, live_key)
                        flat_value_cache.index_copy_(0, prefill_slots, live_value)
                else:
                    block_cursor = 0
                    query_cursor = 0
                    for seq_len, num_blocks, query_len in zip(
                        seq_lens_list, num_blocks_list, query_lens, strict=True
                    ):
                        live_start = seq_len - query_len
                        if query_len > 0:
                            first_block = live_start // block_size
                            last_block = (seq_len - 1) // block_size
                            first_offset = live_start % block_size
                            query_offset = 0

                            if first_block == last_block:
                                row = block_cursor + first_block
                                end_offset = first_offset + query_len
                                q_slice = slice(query_cursor, query_cursor + query_len)
                                key_cache[row, first_offset:end_offset].copy_(key[q_slice])
                                value_cache[row, first_offset:end_offset].copy_(value[q_slice])
                            else:
                                if first_offset != 0:
                                    first_len = block_size - first_offset
                                    row = block_cursor + first_block
                                    q_slice = slice(
                                        query_cursor,
                                        query_cursor + first_len,
                                    )
                                    key_cache[row, first_offset:].copy_(key[q_slice])
                                    value_cache[row, first_offset:].copy_(value[q_slice])
                                    query_offset += first_len
                                    full_block_start = first_block + 1
                                else:
                                    full_block_start = first_block

                                last_len = seq_len % block_size
                                if last_len == 0:
                                    full_block_end = last_block + 1
                                else:
                                    full_block_end = last_block

                                num_full_blocks = full_block_end - full_block_start
                                if num_full_blocks > 0:
                                    full_len = num_full_blocks * block_size
                                    q_slice = slice(
                                        query_cursor + query_offset,
                                        query_cursor + query_offset + full_len,
                                    )
                                    key_cache[
                                        block_cursor
                                        + full_block_start : block_cursor
                                        + full_block_start
                                        + num_full_blocks
                                    ].copy_(
                                        key[q_slice].view(
                                            num_full_blocks,
                                            block_size,
                                            self.num_kv_heads,
                                            self.head_size,
                                        )
                                    )
                                    value_cache[
                                        block_cursor
                                        + full_block_start : block_cursor
                                        + full_block_start
                                        + num_full_blocks
                                    ].copy_(
                                        value[q_slice].view(
                                            num_full_blocks,
                                            block_size,
                                            self.num_kv_heads,
                                            self.head_size,
                                        )
                                    )
                                    query_offset += full_len

                                if last_len != 0:
                                    row = block_cursor + last_block
                                    q_slice = slice(
                                        query_cursor + query_offset,
                                        query_cursor + query_offset + last_len,
                                    )
                                    key_cache[row, :last_len].copy_(key[q_slice])
                                    value_cache[row, :last_len].copy_(value[q_slice])
                        block_cursor += num_blocks
                        query_cursor += query_len
                if not self._enable_prefix_caching and attn_metadata.max_query_len > 1:
                    if full_live_prefill:
                        self._kvfloat13_batched_shadow_generation = layout_generation
                    else:
                        self._set_kvfloat13_batched_shadow_cache(
                            kv_cache,
                            key_cache,
                            value_cache,
                            layout_generation,
                        )

        cu_seqlens_q = attn_metadata.query_start_loc
        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
        q_descale = (
            layer._q_scale.expand(descale_shape)
            if self.supports_quant_query_input
            else None
        )
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        with _nvtx_range("fa.kvfloat13.batched.flash_attn"):
            flash_attn_varlen_func(
                q=query[: attn_metadata.num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[: attn_metadata.num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=compact_block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
        return output

    def _get_kvfloat13_shadow_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_len: int,
        start_pos: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shadow_kv = self._kvfloat13_shadow_kv
        if (
            shadow_kv is None
            or shadow_kv.device != key.device
            or (
                shadow_kv.shape[1] < seq_len
                and self._kvfloat13_allow_dynamic_shadow_growth
            )
        ):
            new_capacity = max(
                seq_len,
                2 * (shadow_kv.shape[1] if shadow_kv is not None else 0),
                256,
            )
            shadow_kv = torch.empty(
                (2, new_capacity, self.num_kv_heads, self.head_size),
                device=key.device,
                dtype=torch.bfloat16,
            )
            self._kvfloat13_shadow_kv = shadow_kv
            self._kvfloat13_shadow_seq_len = 0

        if start_pos == 0:
            shadow_kv[0, :seq_len].copy_(key)
            shadow_kv[1, :seq_len].copy_(value)
            self._kvfloat13_shadow_seq_len = seq_len
        elif start_pos == self._kvfloat13_shadow_seq_len:
            shadow_kv[0, start_pos:seq_len].copy_(key[: seq_len - start_pos])
            shadow_kv[1, start_pos:seq_len].copy_(value[: seq_len - start_pos])
            self._kvfloat13_shadow_seq_len = seq_len
        else:
            num_used_blocks = cdiv(seq_len, block_size)
            used_block_ids = block_table[0, :num_used_blocks]
            decoded_k_shape = (
                num_used_blocks * block_size,
                self.num_kv_heads,
                self.head_size,
            )
            decoded_kv = self._get_kvfloat13_decode_buffer(
                kv_cache,
                num_used_blocks,
            )
            decode_kvfloat13_blocks_triton(
                kv_cache,
                used_block_ids,
                self.head_size,
                out=decoded_kv,
            )
            shadow_kv[0, :seq_len].copy_(decoded_kv[0].reshape(decoded_k_shape)[:seq_len])
            shadow_kv[1, :seq_len].copy_(decoded_kv[1].reshape(decoded_k_shape)[:seq_len])
            self._kvfloat13_shadow_seq_len = seq_len

        return shadow_kv[0, :seq_len], shadow_kv[1, :seq_len]

    def _decode_kvfloat13_blocks(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_size = kv_cache.shape[2]
        if block_table.shape[0] == 1:
            num_used_blocks = cdiv(int(seq_lens[0]), block_size)
            used_block_ids = block_table[0, :num_used_blocks]
            compact_key_cache, compact_value_cache = self._decode_kvfloat13_pair(
                kv_cache,
                used_block_ids,
            )
            compact_block_table = self._get_kvfloat13_compact_block_table(block_table)
            compact_block_table[0, :num_used_blocks] = torch.arange(
                num_used_blocks,
                device=block_table.device,
                dtype=block_table.dtype,
            )
            return compact_key_cache, compact_value_cache, compact_block_table

        if not self._enable_prefix_caching:
            used_block_ids, compact_block_table, _ = build_kvfloat13_row_major_layout(
                block_table,
                seq_lens,
                block_size,
                block_positions=self._get_kvfloat13_block_positions(block_table, seq_lens),
                compact_block_table=self._get_kvfloat13_compact_block_table(
                    block_table
                ),
            )
            compact_key_cache, compact_value_cache = self._decode_kvfloat13_pair(
                kv_cache,
                used_block_ids,
            )
            return compact_key_cache, compact_value_cache, compact_block_table

        block_positions = self._get_kvfloat13_block_positions(block_table, seq_lens)
        num_blocks_per_seq = torch.div(
            seq_lens + block_size - 1,
            block_size,
            rounding_mode="floor",
        )
        valid_mask = block_positions.unsqueeze(0) < num_blocks_per_seq.unsqueeze(1)
        used_block_ids = torch.unique(block_table[valid_mask], sorted=True)

        compact_key_cache, compact_value_cache = self._decode_kvfloat13_pair(
            kv_cache,
            used_block_ids,
        )

        compact_block_table = self._get_kvfloat13_compact_block_table(block_table)
        compact_block_table[valid_mask] = torch.searchsorted(
            used_block_ids,
            block_table[valid_mask].to(torch.long),
        ).to(block_table.dtype)
        return compact_key_cache, compact_value_cache, compact_block_table

    def _decode_kvfloat13_pair(
        self,
        kv_cache: torch.Tensor,
        used_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_used_blocks = int(used_block_ids.numel())
        decoded_kv = self._get_kvfloat13_decode_buffer(
            kv_cache,
            num_used_blocks,
        )
        decode_kvfloat13_blocks_triton(
            kv_cache,
            used_block_ids,
            self.head_size,
            out=decoded_kv,
        )
        return decoded_kv.unbind(0)

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        if is_kvfloat13_kv_cache(self.kv_cache_dtype):
            with _nvtx_range("fa.kv_cache_update.kfloat13"):
                reshape_and_cache_kvfloat13(
                    key,
                    value,
                    kv_cache,
                    slot_mapping,
                )
            return

        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        with _nvtx_range("fa.kv_cache_update.default"):
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def _forward_with_dcp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        block_table: torch.Tensor | None = None,
        q_descale: torch.Tensor | None = None,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        cu_seqlens_q = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        if block_table is None:
            block_table = attn_metadata.block_table

        query = query.contiguous()
        query_across_dcp = get_dcp_group().all_gather(query, dim=1)
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        n = query_across_dcp.shape[0]
        (dcp_context_out,) = current_workspace_manager().get_simultaneous(
            (
                (n, self.num_heads * self.dcp_world_size, self.head_size),
                self._dcp_dtype,
            ),
        )
        context_attn_out, context_lse = flash_attn_varlen_func(
            q=query_across_dcp,
            k=key_cache,
            v=value_cache,
            out=dcp_context_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=attn_metadata.dcp_context_kv_lens,
            max_seqlen_k=attn_metadata.max_dcp_context_kv_len,
            softmax_scale=self.scale,
            causal=False,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        # FA returns LSE in shape [ H, B ] but DCP combine wants [ B, H ]
        context_attn_out_cor, context_lse_cor = self.dcp_combine(
            context_attn_out,
            context_lse.transpose(0, 1),
            get_dcp_group(),
            return_lse=True,
        )
        context_lse_cor = context_lse_cor.transpose(0, 1).contiguous()

        (dcp_query_out,) = current_workspace_manager().get_simultaneous(
            ((query.shape[0], self.num_heads, self.head_size), self._dcp_dtype),
        )
        query_attn_out, query_lse = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=dcp_query_out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_q,
            max_seqlen_k=max_seqlen_q,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            fa_version=self.vllm_flash_attn_version,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=attn_metadata.max_num_splits,
        )
        assert context_attn_out_cor.shape == query_attn_out.shape
        assert context_lse_cor.shape == query_lse.shape
        merge_attn_states(
            output,
            context_attn_out_cor,
            context_lse_cor,
            query_attn_out,
            query_lse,
        )

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        # For encoder attention, process FP8 quantization if needed
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantization is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        cu_seqlens_q = attn_metadata.query_start_loc
        cu_seqlens_k = attn_metadata.query_start_loc
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_query_len

        descale_shape = (
            cu_seqlens_q.shape[0] - 1,  # type: ignore[union-attr]
            self.num_kv_heads,
        )

        # Call flash attention directly on Q, K, V tensors
        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,  # Encoder attention is bidirectional
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=1 if self.batch_invariant_enabled else 0,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    use_local_attention: bool,
    num_sms: int,
    dcp_world_size: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window or use_local_attention:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False
    # disable cascade attention for DCP
    if dcp_world_size > 1:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (
        num_queries_per_kv > 1
        and not use_sliding_window
        and not use_alibi
        and np.all(query_lens == 1)
    )
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (
        num_reqs * num_kv_heads * cdiv(num_queries_per_kv, q_tile_size)
    )
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    max_num_splits: int,
    fa_version: int,
    prefix_scheduler_metadata: torch.Tensor | None = None,
    suffix_scheduler_metadata: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    s_aux: torch.Tensor | None = None,
) -> torch.Tensor:
    assert alibi_slopes is None, "Cascade attention does not support ALiBi."
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window."
    )

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=list(sliding_window),
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        # s_aux is incorporated into prefix_lse inside the GPU kernel,
        # enabling its effect during the final attention merge.
        s_aux=s_aux,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=list(sliding_window),
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
        num_splits=1 if envs.VLLM_BATCH_INVARIANT else max_num_splits,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
