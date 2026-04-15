# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import json
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.models.utils import extract_layer_index
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


def _build_request_history_masks(
    compact_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    num_used_blocks: int,
) -> list[torch.Tensor]:
    masks = []
    seq_lens_list = [int(x) for x in seq_lens.tolist()]
    for req_idx, seq_len in enumerate(seq_lens_list):
        mask = torch.zeros(
            (num_used_blocks, block_size),
            dtype=torch.bool,
            device=compact_block_table.device,
        )
        if seq_len <= 1:
            masks.append(mask)
            continue
        num_blocks = (seq_len + block_size - 1) // block_size
        rows = compact_block_table[req_idx, :num_blocks].to(torch.long).cpu().tolist()
        remaining = seq_len
        for row in rows:
            valid_len = min(block_size, remaining)
            mask[row, :valid_len] = True
            remaining -= valid_len
        live_row = rows[(seq_len - 1) // block_size]
        live_col = (seq_len - 1) % block_size
        mask[live_row, live_col] = False
        masks.append(mask)
    return masks


def _tensor_metrics(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    if current.numel() == 0:
        return {
            "elem_exact_rate": 1.0,
            "token_exact_rate": 1.0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
        }
    exact = current == reference
    diff = (current.float() - reference.float()).abs()
    flat_exact = exact.reshape(exact.shape[0], -1)
    return {
        "elem_exact_rate": float(exact.float().mean().item()),
        "token_exact_rate": float(flat_exact.all(dim=1).float().mean().item()),
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
    }


def _record_request_metrics(
    record_path: Path,
    *,
    layer_name: str,
    decode_step: int,
    req_idx: int,
    seq_len: int,
    kind: str,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    ref_key: torch.Tensor,
    ref_value: torch.Tensor,
) -> None:
    key_metrics = _tensor_metrics(current_key, ref_key)
    value_metrics = _tensor_metrics(current_value, ref_value)
    payload = {
        "layer_name": layer_name,
        "layer_idx": int(extract_layer_index(layer_name)),
        "decode_step": decode_step,
        "request_idx": req_idx,
        "seq_len": seq_len,
        "history_tokens": int(current_key.shape[0]),
        "kind": kind,
        "k": key_metrics,
        "v": value_metrics,
    }
    with record_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def _layerwise_error_patch(record_path: Path) -> Iterator[None]:
    original_forward = flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense
    original_update = flash_attn_backend.FlashAttentionImpl.do_kv_cache_update

    mirror_store: dict[tuple[str, int], torch.Tensor] = {}
    decode_state = {
        "last_seq_lens": None,
        "decode_step": -1,
    }

    def _mirror_key(layer_name: str, kv_cache: torch.Tensor) -> tuple[str, int]:
        return (layer_name, kv_cache.untyped_storage().data_ptr())

    def _get_mirror(
        layer_name: str,
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        key = _mirror_key(layer_name, kv_cache)
        expected_shape = (
            2,
            kv_cache.shape[1],
            kv_cache.shape[2],
            num_kv_heads,
            head_size,
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

    def patched_update(self, layer, key, value, kv_cache, slot_mapping):
        if self.attn_type not in (
            flash_attn_backend.AttentionType.ENCODER_ONLY,
            flash_attn_backend.AttentionType.ENCODER,
        ) and flash_attn_backend.is_kvfloat13_kv_cache(self.kv_cache_dtype):
            mirror = _get_mirror(
                layer.layer_name,
                kv_cache,
                self.num_kv_heads,
                self.head_size,
            )
            _update_bf16_mirror(
                mirror,
                key,
                value,
                slot_mapping,
                slot_mapping.shape[0],
            )
        return original_update(self, layer, key, value, kv_cache, slot_mapping)

    def patched_forward(
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
            attn_metadata.block_table.shape[0] > 1
            and not attn_metadata.use_cascade
            and not self._enable_prefix_caching
            and attn_metadata.max_query_len == 1
        ):
            seq_tuple = tuple(int(x) for x in attn_metadata.seq_lens.tolist())
            if decode_state["last_seq_lens"] != seq_tuple:
                decode_state["last_seq_lens"] = seq_tuple
                decode_state["decode_step"] += 1
            decode_step = int(decode_state["decode_step"])

            block_size = kv_cache.shape[2]
            (
                used_block_ids,
                compact_block_table,
                _seq_lens_list,
                _num_blocks_list,
                _query_lens,
                compact_slots,
                layout_generation,
            ) = self._get_kvfloat13_row_major_cache(
                attn_metadata,
                block_size,
            )
            used_block_ids = used_block_ids.to(torch.long)

            if compact_slots is not None:
                actual_key_cache, actual_value_cache = (
                    self._get_kvfloat13_batched_shadow_cache(
                        kv_cache,
                        used_block_ids,
                        layout_generation,
                    )
                )
            else:
                actual_key_cache, actual_value_cache = self._decode_kvfloat13_pair(
                    kv_cache,
                    used_block_ids,
                )

            decoded_key_cache, decoded_value_cache = self._decode_kvfloat13_pair(
                kv_cache,
                used_block_ids,
            )
            mirror = _get_mirror(
                layer.layer_name,
                kv_cache,
                self.num_kv_heads,
                self.head_size,
            )
            ref_key_cache = mirror[0].index_select(0, used_block_ids)
            ref_value_cache = mirror[1].index_select(0, used_block_ids)
            request_masks = _build_request_history_masks(
                compact_block_table,
                attn_metadata.seq_lens,
                block_size,
                int(used_block_ids.numel()),
            )
            seq_lens_list = [int(x) for x in attn_metadata.seq_lens.tolist()]

            for req_idx, req_mask in enumerate(request_masks):
                history_tokens = int(req_mask.sum().item())
                if history_tokens == 0:
                    continue
                actual_key = actual_key_cache[req_mask]
                actual_value = actual_value_cache[req_mask]
                decoded_key = decoded_key_cache[req_mask]
                decoded_value = decoded_value_cache[req_mask]
                ref_key = ref_key_cache[req_mask]
                ref_value = ref_value_cache[req_mask]
                _record_request_metrics(
                    record_path,
                    layer_name=layer.layer_name,
                    decode_step=decode_step,
                    req_idx=req_idx,
                    seq_len=seq_lens_list[req_idx],
                    kind="actual",
                    current_key=actual_key,
                    current_value=actual_value,
                    ref_key=ref_key,
                    ref_value=ref_value,
                )
                _record_request_metrics(
                    record_path,
                    layer_name=layer.layer_name,
                    decode_step=decode_step,
                    req_idx=req_idx,
                    seq_len=seq_lens_list[req_idx],
                    kind="decoded",
                    current_key=decoded_key,
                    current_value=decoded_value,
                    ref_key=ref_key,
                    ref_value=ref_value,
                )

        return original_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
        )

    flash_attn_backend.FlashAttentionImpl.do_kv_cache_update = patched_update
    flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = patched_forward
    try:
        yield
    finally:
        flash_attn_backend.FlashAttentionImpl.do_kv_cache_update = original_update
        flash_attn_backend.FlashAttentionImpl._forward_kvfloat13_batched_dense = original_forward


def _load_records(record_path: Path) -> list[dict]:
    records = []
    if not record_path.exists():
        return records
    with record_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _nonexact_rate(record: dict) -> float:
    return 1.0 - 0.5 * (
        record["k"]["token_exact_rate"] + record["v"]["token_exact_rate"]
    )


def _step_average(records: list[dict], request_idx: int, kind: str) -> list[dict]:
    per_step: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record["request_idx"] == request_idx and record["kind"] == kind:
            per_step[int(record["decode_step"])].append(record)
    summary = []
    for decode_step, step_records in sorted(per_step.items()):
        summary.append(
            {
                "decode_step": decode_step,
                "avg_nonexact_rate": sum(
                    _nonexact_rate(record) for record in step_records
                )
                / len(step_records),
                "avg_mean_abs": sum(
                    0.5 * (record["k"]["mean_abs"] + record["v"]["mean_abs"])
                    for record in step_records
                )
                / len(step_records),
            }
        )
    return summary


def _top_layers_at_step(
    records: list[dict],
    *,
    request_idx: int,
    decode_step: int,
    kind: str,
    limit: int,
) -> list[dict]:
    step_records = [
        record
        for record in records
        if record["request_idx"] == request_idx
        and record["decode_step"] == decode_step
        and record["kind"] == kind
    ]
    step_records.sort(
        key=lambda record: (
            _nonexact_rate(record),
            0.5 * (record["k"]["mean_abs"] + record["v"]["mean_abs"]),
            0.5 * (record["k"]["max_abs"] + record["v"]["max_abs"]),
        ),
        reverse=True,
    )
    return [
        {
            "layer_name": record["layer_name"],
            "layer_idx": record["layer_idx"],
            "nonexact_rate": _nonexact_rate(record),
            "k_mean_abs": record["k"]["mean_abs"],
            "v_mean_abs": record["v"]["mean_abs"],
            "k_max_abs": record["k"]["max_abs"],
            "v_max_abs": record["v"]["max_abs"],
        }
        for record in step_records[:limit]
    ]


def _summarize_prompt(
    prompt: str,
    request_idx: int,
    bf16_result: RunResult,
    kfloat13_result: RunResult,
    records: list[dict],
    *,
    topk: int,
) -> dict:
    diff_idx = _first_diff(
        bf16_result.token_ids[request_idx],
        kfloat13_result.token_ids[request_idx],
    )
    actual_steps = _step_average(records, request_idx, "actual")
    decoded_steps = _step_average(records, request_idx, "decoded")
    summary = {
        "prompt": prompt,
        "bf16_text": bf16_result.texts[request_idx],
        "kfloat13_text": kfloat13_result.texts[request_idx],
        "match": diff_idx is None,
        "first_diff_token": diff_idx,
        "actual_step_summary": actual_steps,
        "decoded_step_summary": decoded_steps,
    }
    if diff_idx is None:
        return summary

    inspect_steps = []
    if diff_idx > 0:
        inspect_steps.append(diff_idx - 1)
    inspect_steps.append(diff_idx)
    summary["inspect_steps"] = inspect_steps
    summary["top_layers"] = {}
    for decode_step in inspect_steps:
        summary["top_layers"][str(decode_step)] = {
            "actual": _top_layers_at_step(
                records,
                request_idx=request_idx,
                decode_step=decode_step,
                kind="actual",
                limit=topk,
            ),
            "decoded": _top_layers_at_step(
                records,
                request_idx=request_idx,
                decode_step=decode_step,
                kind="decoded",
                limit=topk,
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/root/autodl-tmp/Qwen3-4B",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--attention-backend", type=str, default="FLASH_ATTN")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
    )
    args = parser.parse_args()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    bf16_llm = _make_llm(
        args.model,
        "auto",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        attention_backend=args.attention_backend,
        use_graph=args.use_graph,
    )
    try:
        bf16_outputs = bf16_llm.generate(args.prompts, sampling_params)
        bf16_result = _collect_outputs("bf16", bf16_outputs)
    finally:
        _shutdown_llm(bf16_llm)

    with tempfile.NamedTemporaryFile(
        prefix="kvfloat13-layerwise-",
        suffix=".jsonl",
        delete=False,
    ) as record_file:
        record_path = Path(record_file.name)

    with _layerwise_error_patch(record_path):
        kfloat13_llm = _make_llm(
            args.model,
            "kfloat13",
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            attention_backend=args.attention_backend,
            use_graph=args.use_graph,
        )
        try:
            kfloat13_outputs = kfloat13_llm.generate(args.prompts, sampling_params)
            kfloat13_result = _collect_outputs("kfloat13", kfloat13_outputs)
        finally:
            _shutdown_llm(kfloat13_llm)

    records = _load_records(record_path)
    print(
        json.dumps(
            {
                "record_path": str(record_path),
                "num_records": len(records),
                "use_graph": args.use_graph,
            },
            ensure_ascii=False,
        )
    )
    for request_idx, prompt in enumerate(args.prompts):
        print(
            json.dumps(
                _summarize_prompt(
                    prompt,
                    request_idx,
                    bf16_result,
                    kfloat13_result,
                    records,
                    topk=args.topk,
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
