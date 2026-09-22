# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np

from tools.artifact.layouts import swizzle_nvfp4_scales
from tools.convert.qwen3_8_flash_next.convert import (
    SliceReader,
    _iter_fp8_payload,
    _iter_ple_payload,
    _swizzle_nvfp4_scale_bytes,
)
from tools.convert.qwen3_8_flash_next.recipe import Fp8Recipe, SourceTensor
from tools.convert.qwen3_8_flash_next.source import TensorSlice, read_safetensor_directory


def _write_safetensors(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]]):
    offset = 0
    directory = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        directory[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    header = json.dumps(directory, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_slice_reader_streams_exact_ranges(tmp_path):
    path = tmp_path / "source.safetensors"
    _write_safetensors(path, [("x", "U8", (17,), bytes(range(17)))])
    tensor = read_safetensor_directory(path).tensors["x"]
    with SliceReader() as reader:
        assert b"".join(reader.iter_tensor(tensor, chunk_bytes=5)) == bytes(range(17))


def test_fp8_payload_concatenates_codes_then_f32_row_scales(tmp_path):
    path = tmp_path / "fp8.safetensors"
    codes_a = bytes(range(8))
    codes_b = bytes(range(8, 16))
    scales_a = struct.pack("<2f", 0.5, 1.0)
    scales_b = struct.pack("<2f", 2.0, 4.0)
    _write_safetensors(
        path,
        [
            ("a.weight", "F8_E4M3", (2, 4), codes_a),
            ("a.weight_scale", "F32", (2, 1), scales_a),
            ("b.weight", "F8_E4M3", (2, 4), codes_b),
            ("b.weight_scale", "F32", (2, 1), scales_b),
        ],
    )
    tensors = read_safetensor_directory(path).tensors
    recipe = Fp8Recipe(
        "fixture",
        (
            SourceTensor("a.weight", "F8_E4M3", (2, 4)),
            SourceTensor("b.weight", "F8_E4M3", (2, 4)),
        ),
    )
    with SliceReader() as reader:
        payload = b"".join(
            _iter_fp8_payload(recipe, (4, 4), tensors.__getitem__, reader)
        )
    assert payload[:16] == codes_a + codes_b
    assert payload[16:256] == bytes(240)
    assert payload[256:] == scales_a + scales_b


def test_ple_payload_reorders_source_scale_and_code_planes(tmp_path):
    path = tmp_path / "ple.safetensors"
    codes = bytes(range(16))
    scales = struct.pack("<2e", 0.5, 1.0)
    _write_safetensors(
        path,
        [
            ("weight_scale", "F16", (2, 1), scales),
            ("weight_i4", "U8", (2, 8), codes),
        ],
    )
    tensors = read_safetensor_directory(path).tensors
    with SliceReader() as reader:
        payload = b"".join(_iter_ple_payload((2, 16), tensors, reader))
    assert payload[:16] == codes
    assert payload[16:256] == bytes(240)
    assert payload[256:] == scales


def test_streaming_expert_scale_swizzle_matches_layout_oracle():
    natural = np.arange(512, dtype=np.uint8).reshape(128, 4) & np.uint8(0x7E)
    actual = _swizzle_nvfp4_scale_bytes(natural.tobytes(), 128, 64)

    import torch

    expected = swizzle_nvfp4_scales(
        torch.from_numpy(natural.copy()), (128, 64)
    ).numpy().tobytes()
    assert actual == expected
