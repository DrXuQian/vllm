# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def _shutdown_llm(llm: LLM) -> None:
    llm.llm_engine.engine_core.shutdown()
    del llm
    torch.cuda.empty_cache()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_prompt(tokenizer: AutoTokenizer, target_tokens: int) -> str:
    seed = "The history of artificial intelligence spans several major milestones. "
    ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Tokenizer returned an empty token sequence for the benchmark seed.")

    pieces: list[int] = []
    while len(pieces) < target_tokens:
        pieces.extend(ids)
    pieces = pieces[:target_tokens]
    return tokenizer.decode(pieces, clean_up_tokenization_spaces=False)


def _build_prompts(
    tokenizer: AutoTokenizer,
    prompt_len: int,
    batch_size: int,
) -> tuple[list[str], int]:
    prompts = []
    actual_len = 0
    for _ in range(batch_size):
        prompt = _build_prompt(tokenizer, prompt_len)
        prompts.append(prompt)
        actual_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompts, actual_len


def _make_llm(args: argparse.Namespace, kv_cache_dtype: str) -> LLM:
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=args.attention_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_flashinfer_autotune=False,
    )
    if args.use_graph:
        max_bs = max(_parse_csv_ints(args.batch_sizes))
        llm_kwargs["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": 2,
            "cudagraph_capture_sizes": sorted(set([1] + _parse_csv_ints(args.batch_sizes))),
            "max_cudagraph_capture_size": max_bs,
        }
    else:
        llm_kwargs["enforce_eager"] = True
    return LLM(**llm_kwargs)


def _run_case(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    repeat: int,
) -> tuple[float, list[float], str]:
    llm.generate(["warmup"], SamplingParams(temperature=0.0, max_tokens=2))
    total_output_tokens = len(prompts) * sampling_params.max_tokens
    measurements: list[float] = []
    preview = ""
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        measurements.append(total_output_tokens / elapsed)
        preview = outputs[0].outputs[0].text
    return sum(measurements) / len(measurements), measurements, preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark KVFloat13 against BF16 KV cache.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--kv-dtypes", default="auto,kfloat13")
    parser.add_argument("--prompt-lengths", default="256,2048")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    results = []
    for kv_dtype in _parse_csv_strs(args.kv_dtypes):
        llm = _make_llm(args, kv_dtype)
        try:
            for prompt_len in _parse_csv_ints(args.prompt_lengths):
                for batch_size in _parse_csv_ints(args.batch_sizes):
                    prompts, actual_prompt_len = _build_prompts(tokenizer, prompt_len, batch_size)
                    avg_tps, tps_runs, preview = _run_case(
                        llm,
                        prompts,
                        sampling_params,
                        args.repeat,
                    )
                    row = {
                        "kv_cache_dtype": kv_dtype,
                        "prompt_len_target": prompt_len,
                        "prompt_len_actual": actual_prompt_len,
                        "batch_size": batch_size,
                        "max_tokens": args.max_tokens,
                        "use_graph": args.use_graph,
                        "avg_tokens_per_sec": avg_tps,
                        "tokens_per_sec_runs": tps_runs,
                        "preview": preview,
                    }
                    results.append(row)
                    print(json.dumps(row, ensure_ascii=False))
        finally:
            _shutdown_llm(llm)

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
