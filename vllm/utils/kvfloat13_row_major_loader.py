# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import threading
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_LOAD_LOCK = threading.Lock()
_LOADED = False


def ensure_kvfloat13_row_major_layout_op() -> bool:
    global _LOADED
    if hasattr(torch.ops, "_C_cache_ops") and hasattr(
        torch.ops._C_cache_ops, "build_kvfloat13_row_major_layout"
    ):
        _LOADED = True
        return True
    if _LOADED:
        return hasattr(torch.ops._C_cache_ops, "build_kvfloat13_row_major_layout")
    if not torch.cuda.is_available():
        return False

    with _LOAD_LOCK:
        if hasattr(torch.ops, "_C_cache_ops") and hasattr(
            torch.ops._C_cache_ops, "build_kvfloat13_row_major_layout"
        ):
            _LOADED = True
            return True

        source = Path(__file__).with_name("kvfloat13_row_major_ext.cu")
        build_root = Path(
            os.environ.get(
                "VLLM_KVFLOAT13_EXT_BUILD_ROOT",
                "/root/autodl-tmp/vllm_kvfloat13_ext_build",
            )
        )
        build_dir = build_root / "kvfloat13_row_major"
        tmp_dir = build_root / "tmp"
        build_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TMPDIR", str(tmp_dir))
        tempfile.tempdir = str(tmp_dir)

        load(
            name="kvfloat13_row_major_ext",
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
            build_directory=str(build_dir),
            verbose=False,
            is_python_module=False,
        )
        _LOADED = True
        return hasattr(torch.ops._C_cache_ops, "build_kvfloat13_row_major_layout")
