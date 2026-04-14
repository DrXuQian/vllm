# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.utils.kvfloat13 import (
    DEFAULT_KVFLOAT13_DECOMPRESS_LUT,
    decode_kvfloat13,
    encode_kvfloat13,
    kvfloat13_page_size_bytes,
    kvfloat13_packed_bytes_per_head,
    reshape_and_cache_kvfloat13,
)
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
