# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import json
from dataclasses import dataclass
from typing import Iterator

import torch

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends import flash_attn as flash_attn_backend


DEFAULT_PROMPTS = [
    "请用中文简短介绍人工智能的发展历程。",
    "Write one paragraph about the role of KV cache in transformer decoding.",
    "列出三个适合长上下文评测的任务。",
    "Explain why mixed-precision KV cache can change memory-bandwidth tradeoffs.",
]

_RESET_HISTORY_WINDOW_MIRROR: callable | None = None


@dataclass
class RunResult:
    name: str
    texts: list[str]
    token_ids: list[list[int]]


def _shutdown_llm(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()


def _make_llm(
    model: str,
    kv_cache_dtype: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    attention_backend: str,
    use_graph: bool,
) -> LLM:
    llm_kwargs = dict(
        model=model,
        tensor_parallel_size=1,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=attention_backend,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        enable_flashinfer_autotune=False,
    )
    if use_graph:
        llm_kwargs["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": 2,
            "cudagraph_capture_sizes": [1],
            "max_cudagraph_capture_size": 1,
        }
    else:
        llm_kwargs["enforce_eager"] = True
    return LLM(**llm_kwargs)


def _collect_outputs(name: str, outputs) -> RunResult:
    return RunResult(
        name=name,
        texts=[out.outputs[0].text for out in outputs],
        token_ids=[list(out.outputs[0].token_ids) for out in outputs],
    )


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for idx, (lhs, rhs) in enumerate(zip(a, b, strict=False)):
        if lhs != rhs:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _update_bf16_mirror(
    mirror: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_actual_tokens: int,
) -> None:
    slots = slot_mapping[:num_actual_tokens].to(torch.long)
    valid_mask = slots >= 0
    if not bool(valid_mask.any()):
        return
    slots = slots[valid_mask]
    live_key = key[:num_actual_tokens][valid_mask]
    live_value = value[:num_actual_tokens][valid_mask]
    flat_key = mirror[0].view(-1, mirror.shape[3], mirror.shape[4])
    flat_value = mirror[1].view(-1, mirror.shape[3], mirror.shape[4])
    flat_key.index_copy_(0, slots, live_key)
    flat_value.index_copy_(0, slots, live_value)


@contextlib.contextmanager
def _history_window_patch(
    *,
    tail_blocks: int | None,
) -> Iterator[None]:
    global _RESET_HISTORY_WINDOW_MIRROR
    original = flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense
    mirror_store: dict[tuple[int, int], torch.Tensor] = {}

    def _mirror_key(self, kv_cache: torch.Tensor) -> tuple[int, int]:
        return (id(self), kv_cache.untyped_storage().data_ptr())

    def _get_mirror(self, kv_cache: torch.Tensor) -> torch.Tensor:
        key = _mirror_key(self, kv_cache)
        expected_shape = (
            2,
            kv_cache.shape[1],
            kv_cache.shape[2],
            self.num_kv_heads,
            self.head_size,
        )
        mirror = mirror_store.get(key)
        if (
            mirror is None
            or mirror.device != kv_cache.device
            or tuple(mirror.shape) != expected_shape
        ):
            mirror = torch.zeros(
                expected_shape,
                device=kv_cache.device,
                dtype=torch.bfloat16,
            )
            mirror_store[key] = mirror
        return mirror

    def patched(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        mirror = _get_mirror(self, kv_cache)
        _update_bf16_mirror(
            mirror,
            key,
            value,
            attn_metadata.slot_mapping,
            attn_metadata.num_actual_tokens,
        )

        if (
            attn_metadata.block_table.shape[0] == 1
            or attn_metadata.use_cascade
            or self._enable_prefix_caching
        ):
            return original(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )

        block_size = kv_cache.shape[2]
        (
            used_block_ids,
            compact_block_table,
            seq_lens_list,
            num_blocks_list,
            query_lens,
            compact_slots,
            layout_generation,
        ) = self._get_kvfloat13_row_major_cache(
            attn_metadata,
            block_size,
        )

        if tail_blocks is None:
            key_cache = mirror[0].index_select(0, used_block_ids)
            value_cache = mirror[1].index_select(0, used_block_ids)
        else:
            if compact_slots is not None and not self._enable_prefix_caching:
                key_cache, value_cache = self._get_kvfloat13_batched_shadow_cache(
                    kv_cache,
                    used_block_ids,
                    layout_generation,
                )
            else:
                key_cache, value_cache = self._decode_kvfloat13_pair(
                    kv_cache,
                    used_block_ids,
                )

            if compact_slots is not None:
                flat_key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
                flat_value_cache = value_cache.view(
                    -1,
                    self.num_kv_heads,
                    self.head_size,
                )
                live_key = key[: attn_metadata.num_actual_tokens]
                live_value = value[: attn_metadata.num_actual_tokens]
                flat_key_cache.index_copy_(0, compact_slots, live_key)
                flat_value_cache.index_copy_(0, compact_slots, live_value)
            else:
                block_cursor = 0
                query_cursor = 0
                for seq_len, num_blocks, query_len in zip(
                    seq_lens_list,
                    num_blocks_list,
                    query_lens,
                    strict=True,
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
                            value_cache[row, first_offset:end_offset].copy_(
                                value[q_slice]
                            )
                        else:
                            if first_offset != 0:
                                first_len = block_size - first_offset
                                row = block_cursor + first_block
                                q_slice = slice(query_cursor, query_cursor + first_len)
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

            if tail_blocks > 0:
                for req_idx, num_blocks in enumerate(num_blocks_list):
                    tail = min(tail_blocks, num_blocks)
                    if tail <= 0:
                        continue
                    start = num_blocks - tail
                    physical_rows = attn_metadata.block_table[
                        req_idx,
                        start:num_blocks,
                    ].to(torch.long)
                    compact_rows = compact_block_table[
                        req_idx,
                        start:num_blocks,
                    ].to(torch.long)
                    key_cache.index_copy_(
                        0,
                        compact_rows,
                        mirror[0].index_select(0, physical_rows),
                    )
                    value_cache.index_copy_(
                        0,
                        compact_rows,
                        mirror[1].index_select(0, physical_rows),
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
        flash_attn_backend.flash_attn_varlen_func(
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

    flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = patched
    _RESET_HISTORY_WINDOW_MIRROR = mirror_store.clear
    try:
        yield
    finally:
        flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = (
            original
        )
        _RESET_HISTORY_WINDOW_MIRROR = None


def _run_variant(
    *,
    name: str,
    model: str,
    kv_cache_dtype: str,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    attention_backend: str,
    use_graph: bool,
    prompts: list[str],
    max_tokens: int,
    tail_blocks: int | None = 0,
) -> RunResult:
    llm = _make_llm(
        model,
        kv_cache_dtype,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        attention_backend=attention_backend,
        use_graph=use_graph,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    try:
        llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=2))
        if _RESET_HISTORY_WINDOW_MIRROR is not None:
            _RESET_HISTORY_WINDOW_MIRROR()
        outputs = llm.generate(prompts, sampling_params)
        return _collect_outputs(name, outputs)
    finally:
        _shutdown_llm(llm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how much BF16 history is needed to fix KVFloat13 batch drift."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument(
        "--tail-blocks",
        type=int,
        nargs="*",
        default=[0, 1, 2, 4, 8],
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    results: list[RunResult] = [
        _run_variant(
            name="bf16",
            model=args.model,
            kv_cache_dtype="auto",
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=args.attention_backend,
            use_graph=args.use_graph,
            prompts=prompts,
            max_tokens=args.max_tokens,
        )
    ]

    for tail_blocks in args.tail_blocks:
        if tail_blocks == 0:
            results.append(
                _run_variant(
                    name="kfloat13-current",
                    model=args.model,
                    kv_cache_dtype="kfloat13",
                    dtype=args.dtype,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    attention_backend=args.attention_backend,
                    use_graph=args.use_graph,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                )
            )
            continue

        with _history_window_patch(tail_blocks=tail_blocks):
            results.append(
                _run_variant(
                    name=f"kfloat13-tail-{tail_blocks}",
                    model=args.model,
                    kv_cache_dtype="kfloat13",
                    dtype=args.dtype,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    attention_backend=args.attention_backend,
                    use_graph=args.use_graph,
                    prompts=prompts,
                    max_tokens=args.max_tokens,
                )
            )

    with _history_window_patch(tail_blocks=None):
        results.append(
            _run_variant(
                name="kfloat13-full-bf16-history",
                model=args.model,
                kv_cache_dtype="kfloat13",
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                attention_backend=args.attention_backend,
                use_graph=args.use_graph,
                prompts=prompts,
                max_tokens=args.max_tokens,
            )
        )

    baseline = results[0]
    for prompt_idx, prompt in enumerate(prompts):
        row = {"prompt": prompt, "bf16": baseline.texts[prompt_idx]}
        for result in results[1:]:
            row[result.name] = result.texts[prompt_idx]
            row[f"{result.name}_match"] = (
                result.texts[prompt_idx] == baseline.texts[prompt_idx]
            )
            row[f"{result.name}_first_diff"] = _first_diff(
                baseline.token_ids[prompt_idx],
                result.token_ids[prompt_idx],
            )
        print(json.dumps(row, ensure_ascii=False))

    summary = {
        "use_graph": args.use_graph,
        "variants": {
            result.name: {
                "match_rate": sum(
                    text == baseline.texts[idx]
                    for idx, text in enumerate(result.texts)
                )
                / max(len(prompts), 1)
            }
            for result in results[1:]
        },
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
