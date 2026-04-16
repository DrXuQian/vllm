"""
Standard KVFloat13 benchmark with proper cudagraph config.
Must be run as independent process per case.
"""
import argparse
import time
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--kv-dtype", type=str, default="auto")
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    # Build prompt of target token length
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed = "The quick brown fox jumps over the lazy dog. "
    text = seed * (args.prompt_len // len(tokenizer.encode(seed)) + 1)
    tokens = tokenizer.encode(text)[:args.prompt_len]
    prompt = tokenizer.decode(tokens)
    actual_len = len(tokenizer.encode(prompt))

    print(f"Config: model={args.model}")
    print(f"  kv_dtype={args.kv_dtype}")
    print(f"  prompt_len={args.prompt_len} (actual={actual_len})")
    print(f"  batch_size={args.batch_size}")
    print(f"  max_model_len={args.max_model_len}")
    print(f"  gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"  max_tokens={args.max_tokens}")

    compilation_config = {
        "mode": 0,
        "cudagraph_mode": 2,
        "cudagraph_capture_sizes": sorted(set([1, args.batch_size])),
        "max_cudagraph_capture_size": args.batch_size,
    }

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        compilation_config=compilation_config,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
    )

    prompts = [prompt] * args.batch_size

    # Warmup
    print("\nWarmup...")
    _ = llm.generate(prompts, sampling_params)

    # Benchmark
    print(f"\nBenchmark ({args.repeat} repeats):")
    throughputs = []
    for i in range(args.repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = total_output_tokens / elapsed
        throughputs.append(tps)
        print(f"  Run {i+1}: {tps:.2f} tok/s ({total_output_tokens} tokens in {elapsed:.3f}s)")

    avg = sum(throughputs) / len(throughputs)
    steady = sum(throughputs[1:]) / len(throughputs[1:]) if len(throughputs) > 1 else avg
    print(f"\n  Average: {avg:.2f} tok/s")
    print(f"  Steady-state (excl first): {steady:.2f} tok/s")


if __name__ == "__main__":
    main()
