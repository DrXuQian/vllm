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

Low-level FlashInfer kernel benchmark on the target case
(`batch=4, seq=2048, qh=32, kh=8, head_dim=128, page_size=16`) with CUDA events:

- BF16 mean: `18.915 us`
- KVFloat13 mean: `18.689 us`

End-to-end vLLM benchmark with `Qwen3-4B`, `FLASH_ATTN`, CUDA graph,
`cudagraph_capture_sizes=[1,4]`:

- Short context, `batch=4`, actual prompt len about `241`
  - BF16 steady-state: `250.71 tok/s`
  - KVFloat13 steady-state: `245.04 tok/s`
- Long context, `batch=4`, actual prompt len about `1871`
  - BF16 steady-state: `86.22 tok/s`
  - KVFloat13 steady-state: `83.93 tok/s`

In both cases KVFloat13 still preserves about `+23%` KV capacity versus BF16 in
the current vLLM pipeline.
