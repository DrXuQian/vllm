FlashInfer KVFloat13 Batched Decode Patch
=========================================

This directory contains the final FlashInfer-side patch set used by the
`kvfloat13-cudagraph-v1` branch to improve fused batched decode performance for
KVFloat13.

Files
-----

- `0001-kvfloat13-batched-decode-bf16-staging.patch`
  Unified patch against the FlashInfer headers currently used by vLLM in this
  environment.
- `decode.cuh.modified`
  Full post-patch snapshot of `attention/decode.cuh`.
- `decode_kvfloat13.cuh`
  Full post-patch snapshot of `attention/decode_kvfloat13.cuh`.
- `scheduler.cuh.modified`
  Full post-patch snapshot of `attention/scheduler.cuh`.

Patch intent
------------

The original KVFloat13 batched fused decode path was much slower than BF16 in
the `batch=4, seq=2048` case because the same packed KV chunk was decoded
redundantly by each `ty` row in the batched kernel.

The effective optimization is:

1. Keep global KVFloat13 storage format unchanged.
2. In `BatchDecodeWithPagedKVCacheDevice`, for KVFloat13 only:
   - stage packed K/V chunks into shared memory with `cp.async`
   - decode once into a BF16 shared staging buffer
   - reuse the decoded BF16 tiles across the `ty` rows
3. Dispatch the KVFloat13 batched kernel through a dedicated launch-bounds
   entrypoint.

This reduces repeated integer-heavy decode work without adding any persistent
BF16 history buffer.

How to apply
------------

Apply the unified patch inside a FlashInfer source tree or to the installed
header layout that matches these paths:

```bash
cd <flashinfer-root>
patch -p1 < /path/to/0001-kvfloat13-batched-decode-bf16-staging.patch
```

If you are patching an installed package rather than a git checkout, the
expected include paths are:

- `flashinfer/data/include/flashinfer/attention/decode.cuh`
- `flashinfer/data/include/flashinfer/attention/scheduler.cuh`
- `flashinfer/data/include/flashinfer/attention/decode_kvfloat13.cuh`

Validation summary
------------------

Kernel-only FlashInfer benchmark
--------------------------------

Measured with the installed FlashInfer wrapper on one GPU using CUDA events,
`use_tensor_cores=False`, `qh=32`, `kh=8`, `head_dim=128`, `page_size=16`.
Numbers below are medians:

| batch | seq | BF16 | KVFloat13 | ratio |
|---|---:|---:|---:|---:|
| 1 | 256 | `29.344 us` | `29.744 us` | `1.01x` |
| 1 | 2048 | `30.528 us` | `29.664 us` | `0.97x` |
| 4 | 256 | `28.640 us` | `29.488 us` | `1.03x` |
| 4 | 2048 | `29.776 us` | `35.424 us` | `1.19x` |

The main regression versus the original `2.7x` gap is gone; the remaining
kernel-only gap is concentrated in the `batch=4, seq=2048` point.

End-to-end vLLM FlashInfer benchmark
------------------------------------

Measured with `Qwen3-4B`, `attention_backend='FLASHINFER'`, CUDA graph enabled,
and strictly serial one-case-per-process runs.

Short context uses `prompt_len=256` (actual about `241` tokens).
Long context uses `prompt_len=2048` (actual about `1871` tokens).

| case | BF16 steady-state | KVFloat13 steady-state | ratio |
|---|---:|---:|---:|
| short-b1 | `91.86 tok/s` | `80.11 tok/s` | `0.87x` |
| short-b4 | `256.49 tok/s` | `252.67 tok/s` | `0.98x` |
| long-b1 | `52.12 tok/s` | `55.49 tok/s` | `1.06x` |
| long-b4 | `86.83 tok/s` | `87.16 tok/s` | `1.00x` |

Notes:

- `short-b1` required a runtime headroom fix in vLLM so that `FLASHINFER +
  kfloat13` no longer converts all saved memory into KV blocks and then OOMs
  during warmup or benchmark.
- `short-b4` is now effectively at BF16 parity in the FlashInfer-backed vLLM
  path.
- The remaining short-context deficit is concentrated in `batch=1`, not in the
  batched path.

KV cache capacity
-----------------

`kfloat13` still preserves about `+23%` KV capacity versus BF16 in the current
vLLM pipeline.
