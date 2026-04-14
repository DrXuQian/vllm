# SPDX-License-Identifier: Apache-2.0

"""Fallback stub for environments reusing a prebuilt vLLM extension set.

This worktree runs against the system-installed ``vllm._C`` binary while
experimenting with Python-only KV-cache changes. The matching
``_C_stable_libtorch`` extension is not always present in the installed build,
but the import is unconditional on CUDA platform initialization.

For the Qwen3-4B + KVFloat13 path we only need the main ``vllm._C`` ops, so an
empty module keeps initialization working. If a later code path requires
stable-libtorch custom ops, it will fail when that op is actually invoked.
"""
