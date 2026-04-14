# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import lru_cache

import torch
from vllm.triton_utils import tl, triton

KVFLOAT13_DTYPE_STR = "kfloat13"
KVFLOAT13_CHUNK_SIZE = 128
KVFLOAT13_SIGN_BYTES = 16
KVFLOAT13_EXP_HI_BYTES = 64
KVFLOAT13_EM_BYTES = 128
KVFLOAT13_CHUNK_BYTES = KVFLOAT13_SIGN_BYTES + KVFLOAT13_EXP_HI_BYTES + KVFLOAT13_EM_BYTES

# Default exponent table derived from the user's validated KVFloat13 setup.
# It keeps zero plus the dense exponent band commonly seen in BF16 KV cache.
DEFAULT_KVFLOAT13_DECOMPRESS_LUT = tuple([0] + list(range(101, 132)))


def is_kvfloat13_kv_cache(kv_cache_dtype: str | None) -> bool:
    return kv_cache_dtype == KVFLOAT13_DTYPE_STR


def _validate_head_size(head_size: int) -> None:
    if head_size % KVFLOAT13_CHUNK_SIZE != 0:
        raise ValueError(
            f"KVFloat13 requires head_size to be a multiple of {KVFLOAT13_CHUNK_SIZE}, "
            f"got {head_size}."
        )


def kvfloat13_packed_bytes_per_head(head_size: int) -> int:
    _validate_head_size(head_size)
    return (head_size // KVFLOAT13_CHUNK_SIZE) * KVFLOAT13_CHUNK_BYTES


def kvfloat13_page_size_bytes(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int | None = None,
) -> int:
    if head_size_v is None:
        head_size_v = head_size
    return block_size * num_kv_heads * (
        kvfloat13_packed_bytes_per_head(head_size)
        + kvfloat13_packed_bytes_per_head(head_size_v)
    )


@lru_cache(maxsize=1)
def _default_decompress_lut_cpu() -> torch.Tensor:
    return torch.tensor(DEFAULT_KVFLOAT13_DECOMPRESS_LUT, dtype=torch.uint8)


@lru_cache(maxsize=1)
def _default_compress_lut_cpu() -> torch.Tensor:
    decompress_lut = _default_decompress_lut_cpu().to(torch.int16)
    all_exp = torch.arange(256, dtype=torch.int16).unsqueeze(1)
    nearest = (all_exp - decompress_lut.unsqueeze(0)).abs().argmin(dim=1)
    return nearest.to(torch.uint8)


@lru_cache(maxsize=1)
def _sign_unpack_lut_cpu() -> torch.Tensor:
    values = torch.arange(256, dtype=torch.int32).unsqueeze(1)
    shifts = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    return ((values >> shifts) & 1).to(torch.uint8)


@lru_cache(maxsize=1)
def _nibble_unpack_lut_cpu() -> torch.Tensor:
    values = torch.arange(256, dtype=torch.int32)
    lut = torch.empty((256, 2), dtype=torch.uint8)
    lut[:, 0] = (values & 0x0F).to(torch.uint8)
    lut[:, 1] = ((values >> 4) & 0x0F).to(torch.uint8)
    return lut


@lru_cache(maxsize=1)
def _default_bf16_hi_lut_cpu() -> torch.Tensor:
    exp8 = _default_decompress_lut_cpu().to(torch.int32)
    exp_bits = exp8 << 7
    lut = torch.empty((64,), dtype=torch.int32)
    lut[:32] = exp_bits
    lut[32:] = exp_bits | (1 << 15)
    return lut


def get_default_kvfloat13_luts(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    compress = _default_compress_lut_cpu().to(device=device, non_blocking=True)
    decompress = _default_decompress_lut_cpu().to(device=device, non_blocking=True)
    return compress, decompress


@lru_cache(maxsize=None)
def _default_sign_unpack_lut(device: torch.device) -> torch.Tensor:
    return _sign_unpack_lut_cpu().to(device=device, non_blocking=True)


@lru_cache(maxsize=None)
def _default_nibble_unpack_lut(device: torch.device) -> torch.Tensor:
    return _nibble_unpack_lut_cpu().to(device=device, non_blocking=True)


@lru_cache(maxsize=None)
def _default_bf16_hi_lut(device: torch.device) -> torch.Tensor:
    return _default_bf16_hi_lut_cpu().to(device=device, non_blocking=True)


def _pack_sign_bits(sign: torch.Tensor) -> torch.Tensor:
    sign = sign.reshape(*sign.shape[:-1], KVFLOAT13_SIGN_BYTES, 8).to(torch.int16)
    bit_weights = (1 << torch.arange(8, device=sign.device, dtype=torch.int16)).view(
        *((1,) * (sign.ndim - 1)), 8
    )
    return (sign * bit_weights).sum(dim=-1).to(torch.uint8)


def _unpack_sign_bits(signs_packed: torch.Tensor) -> torch.Tensor:
    lut = _default_sign_unpack_lut(signs_packed.device)
    return lut[signs_packed.to(torch.long)].reshape(
        *signs_packed.shape[:-1], KVFLOAT13_CHUNK_SIZE
    )


def _unpack_exp_hi_nibbles(exp_hi_packed: torch.Tensor) -> torch.Tensor:
    lut = _default_nibble_unpack_lut(exp_hi_packed.device)
    return lut[exp_hi_packed.to(torch.long)].reshape(
        *exp_hi_packed.shape[:-1], KVFLOAT13_CHUNK_SIZE
    )


@triton.jit
def _decode_kvfloat13_kernel(
    packed_ptr,
    out_ptr,
    packed_row_stride: tl.int64,
    out_row_stride: tl.int64,
    CHUNK_BYTES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    EXP_HI_BYTES: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    offs = tl.arange(0, CHUNK_SIZE)
    packed_base = packed_ptr + row_idx * packed_row_stride + chunk_idx * CHUNK_BYTES

    sign_bytes = tl.load(packed_base + (offs // 8)).to(tl.uint16)
    sign = (sign_bytes >> (offs % 8)) & 1

    exp_hi_bytes = tl.load(packed_base + SIGN_BYTES + (offs // 2)).to(tl.uint16)
    exp_hi4 = tl.where(
        (offs & 1) == 0,
        exp_hi_bytes & 0x0F,
        (exp_hi_bytes >> 4) & 0x0F,
    )

    em = tl.load(
        packed_base + SIGN_BYTES + EXP_HI_BYTES + offs
    ).to(tl.uint16)
    exp5 = (exp_hi4 << 1) | (em >> 7)
    exp8 = tl.where(exp5 == 0, 0, exp5 + 100)
    bf16_bits = ((sign << 15) | (exp8 << 7) | (em & 0x7F)).to(tl.uint16)
    bf16_vals = bf16_bits.to(tl.bfloat16, bitcast=True)

    out_ptrs = out_ptr + row_idx * out_row_stride + chunk_idx * CHUNK_SIZE + offs
    tl.store(out_ptrs, bf16_vals)


@triton.jit
def _encode_kvfloat13_kernel(
    in_ptr,
    out_ptr,
    compress_lut_ptr,
    in_row_stride: tl.int64,
    out_row_stride: tl.int64,
    CHUNK_BYTES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    EXP_HI_BYTES: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_group_idx = tl.program_id(1)
    groups_per_chunk = CHUNK_SIZE // 8
    chunk_idx = chunk_group_idx // groups_per_chunk
    group_idx = chunk_group_idx % groups_per_chunk
    group_base = group_idx * 8
    offs = tl.arange(0, 8)

    in_ptrs = in_ptr + row_idx * in_row_stride + chunk_idx * CHUNK_SIZE + group_base + offs
    vals = tl.load(in_ptrs).to(tl.uint16, bitcast=True)
    sign = (vals >> 15) & 1
    exp8 = (vals >> 7) & 0xFF
    mant7 = vals & 0x7F
    exp5 = tl.load(compress_lut_ptr + exp8).to(tl.uint16)

    out_base = out_ptr + row_idx * out_row_stride + chunk_idx * CHUNK_BYTES

    sign_bits = tl.sum((sign << tl.arange(0, 8)).to(tl.uint8), axis=0)
    tl.store(out_base + group_idx, sign_bits)

    exp_hi = exp5 >> 1
    pair_ids = offs // 2
    exp_hi_contrib = tl.where((offs & 1) == 0, exp_hi, exp_hi << 4).to(tl.uint8)
    exp_hi_packed = tl.sum(
        tl.where(
            tl.arange(0, 4)[:, None] == pair_ids[None, :],
            exp_hi_contrib[None, :],
            0,
        ),
        axis=1,
    )
    tl.store(
        out_base + SIGN_BYTES + group_idx * 4 + tl.arange(0, 4),
        exp_hi_packed,
    )

    exp_lo_mant = (((exp5 & 1) << 7) | mant7).to(tl.uint8)
    tl.store(out_base + SIGN_BYTES + EXP_HI_BYTES + group_base + offs, exp_lo_mant)


@triton.jit
def _decode_kvfloat13_gather_kernel(
    packed_ptr,
    used_block_ids_ptr,
    out_ptr,
    packed_stride_kv: tl.int64,
    packed_stride_block: tl.int64,
    packed_stride_slot: tl.int64,
    packed_stride_head: tl.int64,
    out_row_stride: tl.int64,
    num_used_blocks: tl.int64,
    NUM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    EXP_HI_BYTES: tl.constexpr,
):
    row_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    rows_per_kv = num_used_blocks * BLOCK_SIZE * NUM_HEADS
    kv_idx = row_idx // rows_per_kv
    rem = row_idx % rows_per_kv
    local_block_idx = rem // (BLOCK_SIZE * NUM_HEADS)
    rem = rem % (BLOCK_SIZE * NUM_HEADS)
    slot_idx = rem // NUM_HEADS
    head_idx = rem % NUM_HEADS

    physical_block_idx = tl.load(used_block_ids_ptr + local_block_idx).to(tl.int64)
    packed_base = (
        packed_ptr
        + kv_idx * packed_stride_kv
        + physical_block_idx * packed_stride_block
        + slot_idx * packed_stride_slot
        + head_idx * packed_stride_head
        + chunk_idx * CHUNK_BYTES
    )

    offs = tl.arange(0, CHUNK_SIZE)
    sign_bytes = tl.load(packed_base + (offs // 8)).to(tl.uint16)
    sign = (sign_bytes >> (offs % 8)) & 1

    exp_hi_bytes = tl.load(packed_base + SIGN_BYTES + (offs // 2)).to(tl.uint16)
    exp_hi4 = tl.where(
        (offs & 1) == 0,
        exp_hi_bytes & 0x0F,
        (exp_hi_bytes >> 4) & 0x0F,
    )

    em = tl.load(packed_base + SIGN_BYTES + EXP_HI_BYTES + offs).to(tl.uint16)
    exp5 = (exp_hi4 << 1) | (em >> 7)
    exp8 = tl.where(exp5 == 0, 0, exp5 + 100)
    bf16_bits = ((sign << 15) | (exp8 << 7) | (em & 0x7F)).to(tl.uint16)
    bf16_vals = bf16_bits.to(tl.bfloat16, bitcast=True)

    out_ptrs = out_ptr + row_idx * out_row_stride + chunk_idx * CHUNK_SIZE + offs
    tl.store(out_ptrs, bf16_vals)


@triton.jit
def _reshape_and_cache_kvfloat13_kernel(
    key_ptr,
    value_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    compress_lut_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_cache_kv: tl.int64,
    stride_cache_block: tl.int64,
    stride_cache_slot: tl.int64,
    stride_cache_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    CHUNK_BYTES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    SIGN_BYTES: tl.constexpr,
    EXP_HI_BYTES: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    chunk_group_idx = tl.program_id(2)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    block_idx = slot // block_size
    block_offset = slot % block_size

    groups_per_chunk = CHUNK_SIZE // 8
    groups_per_head = head_size // 8
    kv_idx = chunk_group_idx // groups_per_head
    head_group_idx = chunk_group_idx % groups_per_head
    chunk_idx = head_group_idx // groups_per_chunk
    group_idx = head_group_idx % groups_per_chunk
    group_base = group_idx * 8
    offs = tl.arange(0, 8)

    src_ptr = tl.where(
        kv_idx == 0,
        key_ptr + tok * stride_key_tok + head * stride_key_head,
        value_ptr + tok * stride_val_tok + head * stride_val_head,
    )
    vals = tl.load(src_ptr + chunk_idx * CHUNK_SIZE + group_base + offs).to(
        tl.uint16, bitcast=True
    )
    sign = (vals >> 15) & 1
    exp8 = (vals >> 7) & 0xFF
    mant7 = vals & 0x7F
    exp5 = tl.load(compress_lut_ptr + exp8).to(tl.uint16)

    out_base = (
        kv_cache_ptr
        + kv_idx * stride_cache_kv
        + block_idx * stride_cache_block
        + block_offset * stride_cache_slot
        + head * stride_cache_head
        + chunk_idx * CHUNK_BYTES
    )

    sign_bits = tl.sum((sign << tl.arange(0, 8)).to(tl.uint8), axis=0)
    tl.store(out_base + group_idx, sign_bits)

    pair_ids = offs // 2
    exp_hi = exp5 >> 1
    exp_hi_contrib = tl.where((offs & 1) == 0, exp_hi, exp_hi << 4).to(tl.uint8)
    exp_hi_packed = tl.sum(
        tl.where(
            tl.arange(0, 4)[:, None] == pair_ids[None, :],
            exp_hi_contrib[None, :],
            0,
        ),
        axis=1,
    )
    tl.store(
        out_base + SIGN_BYTES + group_idx * 4 + tl.arange(0, 4),
        exp_hi_packed,
    )

    exp_lo_mant = (((exp5 & 1) << 7) | mant7).to(tl.uint8)
    tl.store(out_base + SIGN_BYTES + EXP_HI_BYTES + group_base + offs, exp_lo_mant)


def encode_kvfloat13(
    tensor: torch.Tensor,
    compress_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"KVFloat13 expects BF16 input, got {tensor.dtype}.")
    _validate_head_size(tensor.shape[-1])
    if tensor.is_cuda and compress_lut is None:
        return _encode_kvfloat13_triton(tensor)

    if compress_lut is None:
        compress_lut, _ = get_default_kvfloat13_luts(tensor.device)
    else:
        compress_lut = compress_lut.to(device=tensor.device, dtype=torch.uint8)

    num_chunks = tensor.shape[-1] // KVFLOAT13_CHUNK_SIZE
    orig_shape = tensor.shape[:-1]

    tensor_u16 = tensor.contiguous().view(torch.uint16)
    chunks = tensor_u16.reshape(-1, num_chunks, KVFLOAT13_CHUNK_SIZE).to(torch.int32)

    sign = ((chunks >> 15) & 1).to(torch.uint8)
    exp8 = ((chunks >> 7) & 0xFF).to(torch.long)
    mant7 = (chunks & 0x7F).to(torch.uint8)

    exp5 = compress_lut[exp8]
    exp_hi4 = exp5 >> 1
    exp_lo1 = exp5 & 1

    signs_packed = _pack_sign_bits(sign)
    exp_pairs = exp_hi4.reshape(-1, num_chunks, KVFLOAT13_EXP_HI_BYTES, 2)
    exp_hi_packed = (exp_pairs[..., 0] | (exp_pairs[..., 1] << 4)).to(torch.uint8)
    exp_lo_mant = ((exp_lo1 << 7) | mant7).to(torch.uint8)

    packed = torch.cat((signs_packed, exp_hi_packed, exp_lo_mant), dim=-1)
    return packed.reshape(*orig_shape, num_chunks * KVFLOAT13_CHUNK_BYTES)


def _encode_kvfloat13_triton(tensor: torch.Tensor) -> torch.Tensor:
    compress_lut, _ = get_default_kvfloat13_luts(tensor.device)
    num_chunks = tensor.shape[-1] // KVFLOAT13_CHUNK_SIZE
    orig_shape = tensor.shape[:-1]
    tensor_2d = tensor.contiguous().reshape(-1, tensor.shape[-1])
    out = torch.empty(
        (tensor_2d.shape[0], num_chunks * KVFLOAT13_CHUNK_BYTES),
        device=tensor.device,
        dtype=torch.uint8,
    )
    grid = (tensor_2d.shape[0], num_chunks * (KVFLOAT13_CHUNK_SIZE // 8))
    _encode_kvfloat13_kernel[grid](
        tensor_2d,
        out,
        compress_lut,
        tensor_2d.stride(0),
        out.stride(0),
        CHUNK_BYTES=KVFLOAT13_CHUNK_BYTES,
        CHUNK_SIZE=KVFLOAT13_CHUNK_SIZE,
        SIGN_BYTES=KVFLOAT13_SIGN_BYTES,
        EXP_HI_BYTES=KVFLOAT13_EXP_HI_BYTES,
    )
    return out.reshape(*orig_shape, num_chunks * KVFLOAT13_CHUNK_BYTES)


def _decode_kvfloat13_torch(
    packed: torch.Tensor,
    head_size: int,
    decompress_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"KVFloat13 packed storage must be uint8, got {packed.dtype}.")
    _validate_head_size(head_size)
    expected_packed = kvfloat13_packed_bytes_per_head(head_size)
    if packed.shape[-1] != expected_packed:
        raise ValueError(
            f"Packed KVFloat13 last dimension mismatch: expected {expected_packed}, "
            f"got {packed.shape[-1]}."
        )
    use_default_tables = decompress_lut is None
    if use_default_tables:
        _, decompress_lut = get_default_kvfloat13_luts(packed.device)
        bf16_hi_lut = _default_bf16_hi_lut(packed.device)
    else:
        decompress_lut = decompress_lut.to(device=packed.device, dtype=torch.uint8)
        exp8 = decompress_lut.to(torch.int32)
        exp_bits = exp8 << 7
        bf16_hi_lut = torch.empty((64,), dtype=torch.int32, device=packed.device)
        bf16_hi_lut[:32] = exp_bits
        bf16_hi_lut[32:] = exp_bits | (1 << 15)

    num_chunks = head_size // KVFLOAT13_CHUNK_SIZE
    orig_shape = packed.shape[:-1]
    chunks = packed.contiguous().reshape(-1, num_chunks, KVFLOAT13_CHUNK_BYTES)

    signs_packed = chunks[..., :KVFLOAT13_SIGN_BYTES]
    exp_hi_packed = chunks[
        ..., KVFLOAT13_SIGN_BYTES : KVFLOAT13_SIGN_BYTES + KVFLOAT13_EXP_HI_BYTES
    ]
    exp_lo_mant = chunks[..., KVFLOAT13_SIGN_BYTES + KVFLOAT13_EXP_HI_BYTES :]

    sign = _unpack_sign_bits(signs_packed)
    exp_hi4 = _unpack_exp_hi_nibbles(exp_hi_packed)
    mant7 = (exp_lo_mant & 0x7F).to(torch.int32)
    exp_lo1 = (exp_lo_mant >> 7).to(torch.long)
    exp5 = (exp_hi4.to(torch.long) << 1) | exp_lo1
    bf16_key = exp5 | (sign.to(torch.long) << 5)
    bf16_u16 = (bf16_hi_lut[bf16_key] | mant7).to(torch.uint16)
    return bf16_u16.reshape(*orig_shape, head_size).view(torch.bfloat16)


def _decode_kvfloat13_triton(
    packed: torch.Tensor,
    head_size: int,
) -> torch.Tensor:
    num_chunks = head_size // KVFLOAT13_CHUNK_SIZE
    orig_shape = packed.shape[:-1]
    packed_2d = packed.contiguous().reshape(-1, packed.shape[-1])
    out = torch.empty(
        (packed_2d.shape[0], head_size),
        device=packed.device,
        dtype=torch.bfloat16,
    )
    grid = (packed_2d.shape[0], num_chunks)
    _decode_kvfloat13_kernel[grid](
        packed_2d,
        out,
        packed_2d.stride(0),
        out.stride(0),
        CHUNK_BYTES=KVFLOAT13_CHUNK_BYTES,
        CHUNK_SIZE=KVFLOAT13_CHUNK_SIZE,
        SIGN_BYTES=KVFLOAT13_SIGN_BYTES,
        EXP_HI_BYTES=KVFLOAT13_EXP_HI_BYTES,
    )
    return out.reshape(*orig_shape, head_size)


def decode_kvfloat13_blocks_triton(
    kv_cache: torch.Tensor,
    used_block_ids: torch.Tensor,
    head_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not kv_cache.is_cuda or kv_cache.dtype != torch.uint8:
        raise TypeError("KVFloat13 Triton block decode requires CUDA uint8 cache.")
    if kv_cache.ndim != 5:
        raise ValueError(
            f"Expected kv_cache shape [2, num_blocks, block_size, num_heads, packed], "
            f"got {tuple(kv_cache.shape)}."
        )
    if used_block_ids.device != kv_cache.device:
        used_block_ids = used_block_ids.to(device=kv_cache.device)

    num_heads = kv_cache.shape[3]
    block_size = kv_cache.shape[2]
    num_used_blocks = int(used_block_ids.numel())
    out_shape = (2, num_used_blocks, block_size, num_heads, head_size)
    if out is None:
        out = torch.empty(out_shape, device=kv_cache.device, dtype=torch.bfloat16)
    else:
        if out.shape != out_shape or out.dtype != torch.bfloat16:
            raise ValueError(
                f"Provided output has shape/dtype {tuple(out.shape)}/{out.dtype}, "
                f"expected {out_shape}/torch.bfloat16."
            )

    out_2d = out.view(-1, head_size)
    rows = out_2d.shape[0]
    num_chunks = head_size // KVFLOAT13_CHUNK_SIZE
    grid = (rows, num_chunks)
    _decode_kvfloat13_gather_kernel[grid](
        kv_cache,
        used_block_ids,
        out_2d,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        out_2d.stride(0),
        num_used_blocks,
        NUM_HEADS=num_heads,
        BLOCK_SIZE=block_size,
        CHUNK_BYTES=KVFLOAT13_CHUNK_BYTES,
        CHUNK_SIZE=KVFLOAT13_CHUNK_SIZE,
        SIGN_BYTES=KVFLOAT13_SIGN_BYTES,
        EXP_HI_BYTES=KVFLOAT13_EXP_HI_BYTES,
    )
    return out


def decode_kvfloat13(
    packed: torch.Tensor,
    head_size: int,
    decompress_lut: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        packed.is_cuda
        and packed.dtype == torch.uint8
        and packed.shape[-1] == kvfloat13_packed_bytes_per_head(head_size)
        and decompress_lut is None
    ):
        return _decode_kvfloat13_triton(packed, head_size)
    return _decode_kvfloat13_torch(packed, head_size, decompress_lut)


def reshape_and_cache_kvfloat13(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise TypeError("KVFloat13 currently only supports BF16 key/value tensors.")
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return

    if key.is_cuda:
        _reshape_and_cache_kvfloat13_triton(
            key[:num_tokens],
            value[:num_tokens],
            kv_cache,
            slot_mapping,
        )
        return

    block_size = kv_cache.shape[2]
    slots = slot_mapping.to(torch.int64)
    valid_mask = slots >= 0
    if not bool(torch.any(valid_mask)):
        return
    slots = slots[valid_mask]
    block_idx = torch.div(slots, block_size, rounding_mode="floor")
    block_offset = slots % block_size

    packed_kv = encode_kvfloat13(
        torch.stack((key[:num_tokens][valid_mask], value[:num_tokens][valid_mask]), dim=0)
    )
    kv_cache[:, block_idx, block_offset] = packed_kv


def _reshape_and_cache_kvfloat13_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    compress_lut, _ = get_default_kvfloat13_luts(key.device)
    num_tokens, num_heads, head_size = key.shape
    block_size = kv_cache.shape[2]
    groups_per_head = head_size // 8
    grid = (num_tokens, num_heads, 2 * groups_per_head)
    _reshape_and_cache_kvfloat13_kernel[grid](
        key,
        value,
        kv_cache,
        slot_mapping,
        compress_lut,
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        block_size=block_size,
        head_size=head_size,
        CHUNK_BYTES=KVFLOAT13_CHUNK_BYTES,
        CHUNK_SIZE=KVFLOAT13_CHUNK_SIZE,
        SIGN_BYTES=KVFLOAT13_SIGN_BYTES,
        EXP_HI_BYTES=KVFLOAT13_EXP_HI_BYTES,
    )
