# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.layouts import (
    block_scale_bank_geometry,
    decode_nvfp4_bank_words,
    decode_fp8_row_scaled_f32_words,
    decode_packed_u4_g16_words,
    dequantize_fp8_row_scaled_f32,
    dequantize_packed_u4_g16,
    encode_fp8_row_scaled_f32,
    encode_nvfp4_bank,
    encode_packed_u4_g16,
    encoded_size,
    packed_u4_geometry,
    row_scale_geometry,
)
from tools.artifact.numeric import valid_fp32_row_scale_word


def test_fp8_f32_row_scale_known_words_and_reconstruction():
    shape = (2, 4)
    geometry = row_scale_geometry("FP8_E4M3FN_ROW_F32S", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (8, 256, 8, 264)
    assert encoded_size("row-scale-f32-v1", "FP8_E4M3FN_ROW_F32S", shape) == 264

    codes = torch.tensor(
        [[0x00, 0x80, 0x38, 0xB8], [0x40, 0xC0, 0x7E, 0xFE]],
        dtype=torch.uint8,
    )
    scales = torch.tensor([0.5, 2.0], dtype=torch.float32)
    payload = encode_fp8_row_scaled_f32(codes, scales, shape)
    assert payload[:8] == bytes(codes.reshape(-1).tolist())
    assert payload[8:256] == bytes(248)
    assert payload[256:] == struct.pack("<ff", 0.5, 2.0)

    decoded_codes, decoded_scales = decode_fp8_row_scaled_f32_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(
        dequantize_fp8_row_scaled_f32(payload, shape),
        torch.tensor(
            [[0.0, -0.0, 0.5, -0.5], [4.0, -4.0, 896.0, -896.0]],
            dtype=torch.float32,
        ),
    )


def test_fp8_f32_row_scale_rejects_invalid_scales_and_format_pairings():
    assert valid_fp32_row_scale_word(0x00000000)
    assert valid_fp32_row_scale_word(0x3F800000)
    for word in (0x80000000, 0xBF800000, 0x7F800000, 0x7FC00000, 0xFF800000):
        assert not valid_fp32_row_scale_word(word)

    codes = torch.zeros((1, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="nonnegative finite FP32"):
        encode_fp8_row_scaled_f32(codes, torch.tensor([-1.0]), (1, 2))
    with pytest.raises(ValueError, match="does not accept"):
        encoded_size("row-scale-f32-v1", "FP8_E4M3FN_ROW_BF16S", (1, 2))


def test_packed_u4_g16_known_nibbles_scales_and_reconstruction():
    shape = (1, 32)
    geometry = packed_u4_geometry("U4Z8G16_F16S", shape)
    assert (
        geometry.groups_per_row,
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.payload_bytes,
    ) == (2, 16, 256, 4, 260)
    assert encoded_size("packed-u4-g16-v1", "U4Z8G16_F16S", shape) == 260

    packed = torch.tensor(
        [[0x08, 0x17, 0x26, 0x35, 0x44, 0x53, 0x62, 0x71] * 2],
        dtype=torch.uint8,
    )
    scales = torch.tensor([[0.5, 2.0]], dtype=torch.float16)
    payload = encode_packed_u4_g16(packed, scales, shape)
    assert payload[:16] == bytes(packed.reshape(-1).tolist())
    assert payload[16:256] == bytes(240)

    decoded_codes, decoded_scales = decode_packed_u4_g16_words(payload, shape)
    assert torch.equal(decoded_codes, packed)
    assert torch.equal(decoded_scales, scales)
    expected_group = torch.tensor(
        [0, -8, -1, -7, -2, -6, -3, -5, -4, -4, -5, -3, -6, -2, -7, -1],
        dtype=torch.float32,
    )
    expected = torch.cat((expected_group * 0.5, expected_group * 2.0)).reshape(1, 32)
    assert torch.equal(dequantize_packed_u4_g16(payload, shape), expected)


def test_packed_u4_g16_rejects_bad_shape_and_zero_scale_nonzero_codes():
    with pytest.raises(ValueError, match="divisible by 16"):
        packed_u4_geometry("U4Z8G16_F16S", (1, 15))
    packed = torch.zeros((1, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="zero group scale"):
        encode_packed_u4_g16(packed, torch.zeros((1, 1), dtype=torch.float16), (1, 16))


def test_nvfp4_expert_bank_preserves_each_matrix_divisor_and_scale_swizzle():
    shape = (2, 128, 64)
    geometry = block_scale_bank_geometry("NVFP4", shape)
    assert (
        geometry.code_plane_bytes,
        geometry.scale_plane_offset,
        geometry.scale_plane_bytes,
        geometry.weight_divisor_offset,
        geometry.weight_divisor_bytes,
        geometry.payload_bytes,
    ) == (8192, 8192, 1024, 9216, 8, 9224)
    assert encoded_size("expert-blockscale-k16-m128x4-v1", "NVFP4", shape) == 9224

    codes = torch.arange(2 * 128 * 32, dtype=torch.int64).to(torch.uint8).reshape(2, 128, 32)
    scales = (
        torch.arange(2 * 128 * 4, dtype=torch.int64)
        .remainder(0x7F)
        .to(torch.uint8)
        .reshape(2, 128, 4)
    )
    divisors = torch.tensor([1.5, 2.5], dtype=torch.float32)
    payload = encode_nvfp4_bank(codes, scales, divisors, shape)
    decoded_codes, decoded_scales, decoded_divisors = decode_nvfp4_bank_words(payload, shape)
    assert torch.equal(decoded_codes, codes)
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_divisors, divisors)
