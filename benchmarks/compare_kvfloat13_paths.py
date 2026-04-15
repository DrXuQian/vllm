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


def _build_dense_from_compact(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    compact_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens_list = [int(x) for x in seq_lens.tolist()]
    total_tokens = sum(seq_lens_list)
    device = key_cache.device
    dtype = key_cache.dtype

    dense_key = torch.empty(
        (total_tokens, key_cache.shape[2], key_cache.shape[3]),
        device=device,
        dtype=dtype,
    )
    dense_value = torch.empty_like(dense_key)
    cu_seqlens_k = torch.empty(
        (len(seq_lens_list) + 1,),
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k[0] = 0

    token_cursor = 0
    for req_idx, seq_len in enumerate(seq_lens_list):
        num_blocks = (seq_len + block_size - 1) // block_size
        compact_rows = compact_block_table[req_idx, :num_blocks].to(torch.long)
        req_key = key_cache.index_select(0, compact_rows).reshape(
            num_blocks * block_size,
            key_cache.shape[2],
            key_cache.shape[3],
        )
        req_value = value_cache.index_select(0, compact_rows).reshape(
            num_blocks * block_size,
            value_cache.shape[2],
            value_cache.shape[3],
        )
        dense_key[token_cursor : token_cursor + seq_len].copy_(req_key[:seq_len])
        dense_value[token_cursor : token_cursor + seq_len].copy_(req_value[:seq_len])
        token_cursor += seq_len
        cu_seqlens_k[req_idx + 1] = token_cursor

    return dense_key, dense_value, cu_seqlens_k


@contextlib.contextmanager
def _oracle_dense_kvfloat13_patch() -> Iterator[None]:
    original = flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense

    def oracle_dense(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
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
                        value_cache[row, first_offset:end_offset].copy_(value[q_slice])
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

        dense_key, dense_value, cu_seqlens_k = _build_dense_from_compact(
            key_cache,
            value_cache,
            compact_block_table,
            attn_metadata.seq_lens,
            block_size,
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
            k=dense_key,
            v=dense_value,
            out=output[: attn_metadata.num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
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

    flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = oracle_dense
    try:
        yield
    finally:
        flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = original


@contextlib.contextmanager
def _maybe_patch_oracle_dense(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    with _oracle_dense_kvfloat13_patch():
        yield


def _run_once(
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
    oracle_dense: bool = False,
) -> RunResult:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    with _maybe_patch_oracle_dense(oracle_dense):
        llm = _make_llm(
            model,
            kv_cache_dtype,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            attention_backend=attention_backend,
            use_graph=use_graph,
        )
        try:
            llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=2))
            return _collect_outputs(name, llm.generate(prompts, sampling_params))
        finally:
            _shutdown_llm(llm)


def _first_diff(a: list[int], b: list[int]) -> int | None:
    for idx, (lhs, rhs) in enumerate(zip(a, b, strict=False)):
        if lhs != rhs:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BF16, KVFloat13 fast path, and KVFloat13 oracle dense path."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--prompts-file")
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]

    runs = [
        _run_once(
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
        ),
        _run_once(
            name="kfloat13-fast",
            model=args.model,
            kv_cache_dtype="kfloat13",
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=args.attention_backend,
            use_graph=args.use_graph,
            prompts=prompts,
            max_tokens=args.max_tokens,
        ),
        _run_once(
            name="kfloat13-oracle-dense",
            model=args.model,
            kv_cache_dtype="kfloat13",
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=args.attention_backend,
            use_graph=args.use_graph,
            prompts=prompts,
            max_tokens=args.max_tokens,
            oracle_dense=True,
        ),
    ]

    by_name = {run.name: run for run in runs}
    baseline = by_name["bf16"]
    fast = by_name["kfloat13-fast"]
    oracle = by_name["kfloat13-oracle-dense"]

    fast_mismatches = 0
    oracle_mismatches = 0
    for prompt, bf16_text, bf16_ids, fast_text, fast_ids, oracle_text, oracle_ids in zip(
        prompts,
        baseline.texts,
        baseline.token_ids,
        fast.texts,
        fast.token_ids,
        oracle.texts,
        oracle.token_ids,
        strict=True,
    ):
        fast_diff = _first_diff(bf16_ids, fast_ids)
        oracle_diff = _first_diff(bf16_ids, oracle_ids)
        fast_match = bf16_text == fast_text
        oracle_match = bf16_text == oracle_text
        fast_mismatches += int(not fast_match)
        oracle_mismatches += int(not oracle_match)
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "bf16": bf16_text,
                    "kfloat13_fast": fast_text,
                    "kfloat13_oracle_dense": oracle_text,
                    "fast_match": fast_match,
                    "oracle_match": oracle_match,
                    "fast_first_token_diff": fast_diff,
                    "oracle_first_token_diff": oracle_diff,
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "num_prompts": len(prompts),
        "use_graph": args.use_graph,
        "fast_match_rate": 1.0 - fast_mismatches / max(len(prompts), 1),
        "oracle_match_rate": 1.0 - oracle_mismatches / max(len(prompts), 1),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
