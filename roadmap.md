# KVFloat13 Roadmap

## Current Status

- `kv_cache_dtype="kfloat13"` is wired into the vLLM prototype.
- Qwen3-4B runs end-to-end with `kfloat13` KV cache storage.
- KV cache capacity improves by about `+23%` versus BF16.
- Decode CUDA graph works for the current single-request path.
- Custom CUDA ops exist for:
  - `kvfloat13` cache update
  - `kvfloat13` block decode
- Correctness and microbench coverage exists in `tests/test_kvfloat13.py`.

## Current Priority: FlashInfer Short-Context Fixes

The active issue is no longer the low-level FlashInfer fused decode kernel. The
single-kernel KVFloat13 decode path is already close to BF16. The blocker is the
`FLASHINFER + kfloat13` integration path in vLLM:

- `short-b1` has an abnormal memory failure under the same budget where BF16
  succeeds.
- `short-b4` is slower than BF16 even though the low-level fused kernel is
  already near parity.
- The next work must fix memory accounting / hidden allocations first.

### Execution Order

1. Reproduce and document `short-b1` and `short-b4` memory behavior for
   `FLASHINFER + kfloat13`.
2. Capture memory snapshots around:
   - graph profiling
   - FlashInfer wrapper planning
   - warmup
   - KV cache allocation
3. Identify the concrete source of extra reserved memory on the `kfloat13`
   path.
4. Implement the smallest fix that removes the extra memory usage without
   changing semantics.
5. Re-run `short-b1` and `short-b4` end-to-end benchmarks.
6. Only after memory behavior matches expectations, continue optimizing
   `short-b4` throughput.

### Guardrails

- Do not accept fixes that increase permanent BF16 shadow memory.
- Do not accept fixes that trade away the KV cache memory reduction.
- Do not keep microbench-only improvements that fail end-to-end benchmarks.

## Near-Term Goals

1. Stabilize single-request graph performance.
   - Keep the current decode-buffer-only path.
   - Avoid reintroducing full BF16 shadow history.
   - Re-run short/long context graph benchmarks after each kernel change.

2. Improve batched performance without changing semantics.
   - Reduce `row_major_layout` overhead.
   - Reduce `shadow_or_decode` overhead.
   - Keep batched correctness checks alongside performance work.

3. Tighten measurement and diagnostics.
   - Keep `nsys + NVTX` profiling scripts working.
   - Maintain path-comparison and layer-wise error analysis tools.
   - Track `BF16` vs `kfloat13` on short/long context and `batch=1/4`.

## Serving Features To Add

1. Prefix cache compatibility for packed `kfloat13` KV blocks.
2. Swap/offload compatibility for packed KV pages.
3. Cleaner cache-manager integration for packed-byte KV storage.

## Research / Product Extensions

1. K/V asymmetric precision experiments.
2. Layer-wise mixed precision after the uniform `kfloat13` path is stable.
3. Policy/profiler tooling that recommends KV precision by layer.

## Resume Rule

When optimizing, prefer this order:

1. correctness
2. graph-safe execution
3. memory behavior and KV budget correctness
4. batched throughput
4. new features
