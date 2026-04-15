# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm import _custom_ops as ops
from vllm.utils.kvfloat13 import (
    DEFAULT_KVFLOAT13_DECOMPRESS_LUT,
    build_kvfloat13_row_major_layout,
    decode_kvfloat13,
    decode_kvfloat13_blocks_triton,
    encode_kvfloat13,
    get_default_kvfloat13_luts,
    kvfloat13_page_size_bytes,
    kvfloat13_packed_bytes_per_head,
    reshape_and_cache_kvfloat13,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_backend
from vllm.v1.kv_cache_interface import FullAttentionSpec


def _bf16_from_fields(sign: torch.Tensor, exp8: torch.Tensor, mant7: torch.Tensor) -> torch.Tensor:
    raw = (
        (sign.to(torch.int32) << 15)
        | (exp8.to(torch.int32) << 7)
        | mant7.to(torch.int32)
    ).to(torch.uint16)
    return raw.view(torch.bfloat16)


def test_kvfloat13_roundtrip_on_supported_exponents():
    supported_exponents = torch.tensor(DEFAULT_KVFLOAT13_DECOMPRESS_LUT[1:17], dtype=torch.uint16)
    sign = (torch.arange(128, dtype=torch.int32) % 2).to(torch.uint16)
    exp8 = supported_exponents.repeat(8)
    mant7 = (torch.arange(128, dtype=torch.int32) % 128).to(torch.uint16)
    bf16 = _bf16_from_fields(sign, exp8, mant7).reshape(1, 128)

    packed = encode_kvfloat13(bf16)
    decoded = decode_kvfloat13(packed, head_size=128)

    assert packed.shape == (1, 208)
    assert decoded.dtype == torch.bfloat16
    assert torch.equal(decoded.view(torch.uint16), bf16.view(torch.uint16))


def test_kvfloat13_default_lut_matches_validated_boundary_mapping():
    compress_lut, decompress_lut = get_default_kvfloat13_luts(torch.device("cpu"))

    assert decompress_lut.tolist() == list(DEFAULT_KVFLOAT13_DECOMPRESS_LUT)
    assert int(compress_lut[100].item()) == 0
    assert int(compress_lut[101].item()) == 1
    assert int(compress_lut[131].item()) == 31
    assert int(compress_lut[132].item()) == 31


def test_kvfloat13_page_size_and_flash_shape():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        cache_dtype_str="kfloat13",
    )

    expected_page_size = kvfloat13_page_size_bytes(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
    )
    assert spec.page_size_bytes == expected_page_size
    assert expected_page_size == 16 * 8 * 208 * 2

    packed_head_bytes = kvfloat13_packed_bytes_per_head(128)
    assert packed_head_bytes == 208


def test_reshape_and_cache_kvfloat13_writes_expected_slots():
    num_blocks = 2
    block_size = 4
    num_kv_heads = 2
    head_size = 128
    packed_head = kvfloat13_packed_bytes_per_head(head_size)

    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, packed_head), dtype=torch.uint8
    )

    sign = torch.zeros((3, num_kv_heads, head_size), dtype=torch.uint16)
    exp8 = torch.full((3, num_kv_heads, head_size), 124, dtype=torch.uint16)
    mant7 = (
        torch.arange(3 * num_kv_heads * head_size, dtype=torch.int32).reshape(
            3, num_kv_heads, head_size
        )
        % 128
    ).to(torch.uint16)
    key = _bf16_from_fields(sign, exp8, mant7)
    value_mant7 = ((mant7.to(torch.int32) + 7) % 128).to(torch.uint16)
    value = _bf16_from_fields(sign, exp8, value_mant7)
    slot_mapping = torch.tensor([0, 3, 5], dtype=torch.int64)

    reshape_and_cache_kvfloat13(key, value, kv_cache, slot_mapping)

    decoded_key = decode_kvfloat13(kv_cache[0, 0, 0], head_size=head_size)
    decoded_value = decode_kvfloat13(kv_cache[1, 1, 1], head_size=head_size)

    assert torch.equal(decoded_key.view(torch.uint16), key[0].view(torch.uint16))
    assert torch.equal(decoded_value.view(torch.uint16), value[2].view(torch.uint16))


def test_reshape_and_cache_kvfloat13_ignores_negative_slot_mappings():
    num_blocks = 2
    block_size = 4
    num_kv_heads = 2
    head_size = 128
    packed_head = kvfloat13_packed_bytes_per_head(head_size)

    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, packed_head), dtype=torch.uint8
    )

    sign = torch.zeros((3, num_kv_heads, head_size), dtype=torch.uint16)
    exp8 = torch.full((3, num_kv_heads, head_size), 124, dtype=torch.uint16)
    mant7 = (
        torch.arange(3 * num_kv_heads * head_size, dtype=torch.int32).reshape(
            3, num_kv_heads, head_size
        )
        % 128
    ).to(torch.uint16)
    key = _bf16_from_fields(sign, exp8, mant7)
    value_mant7 = ((mant7.to(torch.int32) + 11) % 128).to(torch.uint16)
    value = _bf16_from_fields(sign, exp8, value_mant7)
    slot_mapping = torch.tensor([0, -1, 5], dtype=torch.int64)

    reshape_and_cache_kvfloat13(key, value, kv_cache, slot_mapping)

    decoded_first_key = decode_kvfloat13(kv_cache[0, 0, 0], head_size=head_size)
    decoded_last_value = decode_kvfloat13(kv_cache[1, 1, 1], head_size=head_size)
    untouched_middle = kv_cache[:, 0, 1]

    assert torch.equal(decoded_first_key.view(torch.uint16), key[0].view(torch.uint16))
    assert torch.equal(decoded_last_value.view(torch.uint16), value[2].view(torch.uint16))
    assert torch.count_nonzero(untouched_middle) == 0


def test_build_kvfloat13_row_major_layout_preserves_row_major_order():
    block_table = torch.tensor(
        [
            [7, 9, 0, 0],
            [3, 5, 11, 0],
            [4, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([8, 12, 4], dtype=torch.int32)

    used_block_ids, compact_block_table, compact_slots = build_kvfloat13_row_major_layout(
        block_table,
        seq_lens,
        block_size=4,
    )

    assert torch.equal(used_block_ids, torch.tensor([7, 9, 3, 5, 11, 4], dtype=torch.int32))
    assert torch.equal(
        compact_block_table,
        torch.tensor(
            [
                [0, 1, 0, 0],
                [2, 3, 4, 0],
                [5, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    assert compact_slots is None


def test_build_kvfloat13_row_major_layout_cuda_decode_only_matches_expected_slots():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [7, 9, 0, 0],
            [3, 5, 11, 0],
            [4, 0, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.tensor([8, 12, 4], dtype=torch.int32, device=device)

    used_block_ids, compact_block_table, compact_slots = build_kvfloat13_row_major_layout(
        block_table,
        seq_lens,
        block_size=4,
        decode_only=True,
    )

    assert torch.equal(
        used_block_ids.cpu(),
        torch.tensor([7, 9, 3, 5, 11, 4], dtype=torch.int32),
    )
    assert torch.equal(
        compact_block_table.cpu(),
        torch.tensor(
            [
                [0, 1, 0, 0],
                [2, 3, 4, 0],
                [5, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        compact_slots.cpu(),
        torch.tensor([7, 19, 23], dtype=torch.int64),
    )


def test_kvfloat13_live_suffix_patch_matches_index_copy():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    slots = torch.tensor([5, 1, 7], device=device, dtype=torch.int64)
    num_slots = 9
    num_kv_heads = 3
    head_size = 128

    key_cache = torch.randn(
        num_slots,
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    value_cache = torch.randn_like(key_cache)
    key = torch.randn(
        slots.numel(),
        num_kv_heads,
        head_size,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    ref_key_cache.index_copy_(0, slots, key)
    ref_value_cache.index_copy_(0, slots, value)

    ops.kvfloat13_live_suffix_patch(key_cache, value_cache, slots, key, value)

    assert torch.equal(key_cache.cpu().view(torch.uint16), ref_key_cache.cpu().view(torch.uint16))
    assert torch.equal(
        value_cache.cpu().view(torch.uint16),
        ref_value_cache.cpu().view(torch.uint16),
    )


def test_decode_kvfloat13_blocks_matches_expected_tokens():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    num_blocks = 3
    block_size = 2
    num_kv_heads = 2
    head_size = 128
    packed_head = kvfloat13_packed_bytes_per_head(head_size)

    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, packed_head),
        device=device,
        dtype=torch.uint8,
    )
    num_tokens = num_blocks * block_size
    sign = torch.zeros((num_tokens, num_kv_heads, head_size), device=device, dtype=torch.uint16)
    exp8 = torch.full((num_tokens, num_kv_heads, head_size), 124, device=device, dtype=torch.uint16)
    mant7 = (
        torch.arange(num_tokens * num_kv_heads * head_size, device=device, dtype=torch.int32)
        .reshape(num_tokens, num_kv_heads, head_size)
        % 128
    ).to(torch.uint16)
    key = _bf16_from_fields(sign, exp8, mant7)
    value = _bf16_from_fields(sign, exp8, ((mant7.to(torch.int32) + 17) % 128).to(torch.uint16))
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)

    reshape_and_cache_kvfloat13(key, value, kv_cache, slot_mapping)

    used_block_ids = torch.tensor([2, 0], device=device, dtype=torch.int64)
    decoded = decode_kvfloat13_blocks_triton(kv_cache, used_block_ids, head_size=head_size)

    expected_tokens = torch.cat(
        (
            key[used_block_ids[0] * block_size : (used_block_ids[0] + 1) * block_size],
            key[used_block_ids[1] * block_size : (used_block_ids[1] + 1) * block_size],
        ),
        dim=0,
    )
    expected_values = torch.cat(
        (
            value[used_block_ids[0] * block_size : (used_block_ids[0] + 1) * block_size],
            value[used_block_ids[1] * block_size : (used_block_ids[1] + 1) * block_size],
        ),
        dim=0,
    )

    assert torch.equal(
        decoded[0].reshape(-1, num_kv_heads, head_size).view(torch.uint16).cpu(),
        expected_tokens.view(torch.uint16).cpu(),
    )
    assert torch.equal(
        decoded[1].reshape(-1, num_kv_heads, head_size).view(torch.uint16).cpu(),
        expected_values.view(torch.uint16).cpu(),
    )


def test_kvfloat13_single_request_path_uses_decode_buffer_not_shadow(monkeypatch):
    impl = flash_attn_backend.FlashAttentionImpl(
        num_heads=2,
        head_size=128,
        scale=1.0,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="kfloat13",
    )

    kv_cache = torch.zeros(
        (2, 2, 4, 2, kvfloat13_packed_bytes_per_head(128)),
        dtype=torch.uint8,
    )
    query = torch.randn((1, 2, 128), dtype=torch.bfloat16)
    key = torch.randn((1, 2, 128), dtype=torch.bfloat16)
    value = torch.randn((1, 2, 128), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    attn_metadata = SimpleNamespace(
        num_actual_tokens=1,
        max_seq_len=5,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        max_query_len=1,
        scheduler_metadata=None,
        causal=True,
        max_num_splits=0,
    )
    layer = SimpleNamespace(
        _q_scale=torch.tensor(1.0, dtype=torch.float32),
        _k_scale=torch.tensor(1.0, dtype=torch.float32),
        _v_scale=torch.tensor(1.0, dtype=torch.float32),
    )

    def fail_shadow(*args, **kwargs):
        raise AssertionError("single-request path should not use full BF16 shadow")

    def fake_decode(kv_cache, used_block_ids, head_size, out):
        out[0].fill_(1)
        out[1].fill_(2)

    captured = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured["k"] = kwargs["k"].clone()
        captured["v"] = kwargs["v"].clone()
        kwargs["out"].zero_()

    monkeypatch.setattr(impl, "_get_kvfloat13_shadow_cache", fail_shadow)
    monkeypatch.setattr(
        flash_attn_backend,
        "decode_kvfloat13_blocks_triton",
        fake_decode,
    )
    monkeypatch.setattr(
        flash_attn_backend,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )

    impl._forward_kvfloat13_single_request(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
    )

    assert "k" in captured and "v" in captured
    assert captured["k"].shape == (5, 2, 128)
    assert captured["v"].shape == (5, 2, 128)
    assert torch.all(captured["k"][:4] == 1)
    assert torch.all(captured["v"][:4] == 2)
    assert torch.equal(captured["k"][4].view(torch.uint16), key[0].view(torch.uint16))
    assert torch.equal(captured["v"][4].view(torch.uint16), value[0].view(torch.uint16))
