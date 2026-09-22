# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Revision-pinned source contract for the Qwen3.8-Flash-Next artifact.

The selected conversion source is Primitive's mixed NVFP4/FP8 backbone with
its BF16 PLE shards omitted, plus the companion INT4 PLE directory.  This
module validates the complete splice without loading large tensors into RAM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import BinaryIO, Mapping, Sequence


MIXED_REVISION = "a4e813ed3cfbbcc61e2929699eccb864a4dfa843"
PLE_REVISION = "da8b39586016d8325ac619be28ad77d6296625ec"
MIXED_INDEX_TENSORS = 296_630
EXPERT_LAYERS = 48
EXPERTS = 512
PLE_SHARDS = 128
PLE_ROWS_PER_SHARD = 2_500_012
PLE_ROWS = 320_001_536
PLE_WIDTH = 160

_EXPERT_RE = re.compile(
    r"model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<field>weight_packed|weight_scale|weight_global_scale|input_global_scale)"
)
_PLE_BF16_RE = re.compile(
    r"model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\."
    r"shard_(?P<shard>\d+)\.weight"
)


class SourceError(ValueError):
    """The selected local source does not satisfy the pinned contract."""


@dataclass(frozen=True, slots=True)
class TensorSlice:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    absolute_offset: int
    bytes: int


@dataclass(frozen=True, slots=True)
class SafeTensorDirectory:
    path: Path
    header_bytes: int
    tensors: Mapping[str, TensorSlice]


@dataclass(frozen=True, slots=True)
class BundlePreflight:
    mixed_dir: Path
    ple_dir: Path
    index_tensors: int
    mixed_safetensors: int
    expert_tensors: int
    ple_shards: int
    ple_rows: int
    resource_sha256: Mapping[str, str]


def read_safetensor_directory(path: str | Path) -> SafeTensorDirectory:
    source = Path(path)
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SourceError(f"safetensors header is truncated: {source}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes == 0 or header_bytes > source.stat().st_size - 8:
            raise SourceError(f"invalid safetensors header length: {source}")
        raw_header = handle.read(header_bytes)
    try:
        directory = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"invalid safetensors JSON header: {source}") from error
    if not isinstance(directory, dict):
        raise SourceError(f"safetensors directory is not an object: {source}")

    data_begin = 8 + header_bytes
    tensors: dict[str, TensorSlice] = {}
    max_end = 0
    for name, entry in directory.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise SourceError(f"invalid safetensors entry in {source}")
        try:
            dtype = entry["dtype"]
            shape = tuple(entry["shape"])
            begin, end = entry["data_offsets"]
        except (KeyError, TypeError, ValueError) as error:
            raise SourceError(f"invalid tensor entry {name!r} in {source}") from error
        if (
            not isinstance(dtype, str)
            or not shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end <= begin
        ):
            raise SourceError(f"invalid tensor metadata for {name!r} in {source}")
        tensors[name] = TensorSlice(source, dtype, shape, data_begin + begin, end - begin)
        max_end = max(max_end, end)
    if data_begin + max_end != source.stat().st_size:
        raise SourceError(f"safetensors payload size mismatch: {source}")
    return SafeTensorDirectory(source, header_bytes, tensors)


def _require_revision(local_dir: Path, relative: str, expected: str) -> None:
    metadata = local_dir / ".cache" / "huggingface" / "download" / f"{relative}.metadata"
    if not metadata.is_file():
        raise SourceError(f"Hugging Face revision metadata is missing: {metadata}")
    lines = metadata.read_text(encoding="utf-8").splitlines()
    actual = lines[0] if lines else ""
    if actual != expected:
        raise SourceError(f"source revision mismatch for {relative}: expected {expected}, got {actual}")


def _read_fp32_word(tensor: TensorSlice, handle: BinaryIO) -> int:
    if tensor.dtype != "F32" or tensor.shape != (1,) or tensor.bytes != 4:
        raise SourceError(f"expected one FP32 word: {tensor.path}")
    handle.seek(tensor.absolute_offset)
    raw = handle.read(4)
    if len(raw) != 4:
        raise SourceError(f"truncated FP32 source word: {tensor.path}")
    return struct.unpack("<I", raw)[0]


def _validate_expert_shard(path: Path, layer: int) -> int:
    directory = read_safetensor_directory(path)
    expected_count = EXPERTS * 3 * 4
    if len(directory.tensors) != expected_count:
        raise SourceError(
            f"expert shard {layer} has {len(directory.tensors)} tensors; expected {expected_count}"
        )
    seen: set[tuple[int, str, str]] = set()
    globals_by_expert: dict[int, dict[str, int]] = {}
    with path.open("rb") as handle:
        for name, tensor in directory.tensors.items():
            match = _EXPERT_RE.fullmatch(name)
            if match is None or int(match["layer"]) != layer:
                raise SourceError(f"unexpected expert tensor in layer {layer}: {name}")
            expert = int(match["expert"])
            projection = match["projection"]
            field = match["field"]
            if not 0 <= expert < EXPERTS or (expert, projection, field) in seen:
                raise SourceError(f"invalid or duplicate expert tensor: {name}")
            seen.add((expert, projection, field))
            n, k = (2560, 640) if projection == "down_proj" else (640, 2560)
            expected = {
                "weight_packed": ("U8", (n, k // 2), n * k // 2),
                "weight_scale": ("F8_E4M3", (n, k // 16), n * k // 16),
                "weight_global_scale": ("F32", (1,), 4),
                "input_global_scale": ("F32", (1,), 4),
            }[field]
            if (tensor.dtype, tensor.shape, tensor.bytes) != expected:
                raise SourceError(
                    f"expert tensor signature mismatch for {name}: "
                    f"got {(tensor.dtype, tensor.shape, tensor.bytes)}, expected {expected}"
                )
            if field == "input_global_scale" and _read_fp32_word(tensor, handle) != 0x3F800000:
                raise SourceError(f"expert input scale is not exactly 1.0: {name}")
            if field == "weight_global_scale":
                word = _read_fp32_word(tensor, handle)
                value = struct.unpack("<f", struct.pack("<I", word))[0]
                if not math.isfinite(value) or value <= 0:
                    raise SourceError(f"expert weight divisor is not finite and positive: {name}")
                globals_by_expert.setdefault(expert, {})[projection] = word
    if len(seen) != expected_count:
        raise SourceError(f"expert shard {layer} is incomplete")
    for expert, divisors in globals_by_expert.items():
        if divisors["gate_proj"] != divisors["up_proj"]:
            raise SourceError(
                f"gate/up divisors differ for layer {layer} expert {expert}; cannot fuse the bank"
            )
    return len(seen)


def _validate_config(path: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "hc_count": 4,
        "ngram_vocab_size_base": 20_000_000,
        "split_ngram_parts": 128,
        "ple_embed_dim": 2560,
        "vocab_size": 248_320,
        "bos_token_id": 248_044,
        "eos_token_id": 248_044,
    }
    if config.get("model_type") != "qwen4_exp":
        raise SourceError("mixed config model_type is not qwen4_exp")
    for key, value in expected.items():
        if text.get(key) != value:
            raise SourceError(f"mixed config {key} mismatch: expected {value}, got {text.get(key)}")


def _validate_ple(ple_dir: Path) -> tuple[int, int]:
    meta_path = ple_dir / "META.json"
    _require_revision(ple_dir.parent, "ples_int4/META.json", PLE_REVISION)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {
        "layout": "group16_int4_fp16scale_lownibblefirst",
        "shards": PLE_SHARDS,
        "rows": PLE_ROWS,
        "width": PLE_WIDTH,
    }
    for key, value in required.items():
        if meta.get(key) != value:
            raise SourceError(f"PLE META {key} mismatch: expected {value}, got {meta.get(key)}")

    rows = 0
    for shard in range(PLE_SHARDS):
        path = ple_dir / f"shard_{shard}.safetensors"
        directory = read_safetensor_directory(path)
        if set(directory.tensors) != {"weight_i4", "weight_scale"}:
            raise SourceError(f"PLE shard {shard} has an unexpected tensor inventory")
        codes = directory.tensors["weight_i4"]
        scales = directory.tensors["weight_scale"]
        expected_codes = ("U8", (PLE_ROWS_PER_SHARD, PLE_WIDTH // 2), 200_000_960)
        expected_scales = ("F16", (PLE_ROWS_PER_SHARD, PLE_WIDTH // 16), 50_000_240)
        if (codes.dtype, codes.shape, codes.bytes) != expected_codes:
            raise SourceError(f"PLE code signature mismatch in shard {shard}")
        if (scales.dtype, scales.shape, scales.bytes) != expected_scales:
            raise SourceError(f"PLE scale signature mismatch in shard {shard}")
        rows += PLE_ROWS_PER_SHARD
    return PLE_SHARDS, rows


def validate_bundle(mixed_dir: str | Path, ple_dir: str | Path) -> BundlePreflight:
    mixed = Path(mixed_dir)
    ple_root = Path(ple_dir)
    ple = ple_root / "ples_int4" if (ple_root / "ples_int4").is_dir() else ple_root
    _require_revision(mixed, "config.json", MIXED_REVISION)
    _validate_config(mixed / "config.json")

    index = json.loads((mixed / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or len(weight_map) != MIXED_INDEX_TENSORS:
        raise SourceError(
            f"mixed index has {len(weight_map) if isinstance(weight_map, dict) else 'invalid'} "
            f"tensors; expected {MIXED_INDEX_TENSORS}"
        )

    headers: dict[str, SafeTensorDirectory] = {}
    for path in sorted(mixed.glob("*.safetensors")):
        headers[path.name] = read_safetensor_directory(path)
    expected_files = {
        *(f"ct-experts-layer{layer:02d}.safetensors" for layer in range(EXPERT_LAYERS)),
        "carry-model-bf16-00001.safetensors",
        "carry-model-bf16-00010.safetensors",
        "carry-model-bf16-00011.safetensors",
        "carry-model-bf16-00012.safetensors",
        "fp8-tail-00.safetensors",
    }
    if set(headers) != expected_files:
        raise SourceError("mixed safetensors file inventory differs from the selected bundle")

    omitted_ple: set[int] = set()
    for name, shard in weight_map.items():
        match = _PLE_BF16_RE.fullmatch(name)
        if match is not None:
            omitted_ple.add(int(match["shard"]))
            continue
        if shard not in headers or name not in headers[shard].tensors:
            raise SourceError(f"mixed index tensor is unavailable: {name} -> {shard}")
    if omitted_ple != set(range(PLE_SHARDS)):
        raise SourceError("mixed index does not contain exactly the 128 replaceable BF16 PLE shards")

    expert_tensors = sum(
        _validate_expert_shard(mixed / f"ct-experts-layer{layer:02d}.safetensors", layer)
        for layer in range(EXPERT_LAYERS)
    )
    ple_shards, ple_rows = _validate_ple(ple)

    resources = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    hashes = {
        name: hashlib.sha256((mixed / name).read_bytes()).hexdigest() for name in resources
    }
    return BundlePreflight(
        mixed_dir=mixed,
        ple_dir=ple,
        index_tensors=len(weight_map),
        mixed_safetensors=len(headers),
        expert_tensors=expert_tensors,
        ple_shards=ple_shards,
        ple_rows=ple_rows,
        resource_sha256=hashes,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--ple", required=True)
    args = parser.parse_args(argv)
    result = validate_bundle(args.mixed, args.ple)
    print(
        json.dumps(
            {
                "mixed_dir": str(result.mixed_dir),
                "ple_dir": str(result.ple_dir),
                "index_tensors": result.index_tensors,
                "mixed_safetensors": result.mixed_safetensors,
                "expert_tensors": result.expert_tensors,
                "ple_shards": result.ple_shards,
                "ple_rows": result.ple_rows,
                "resource_sha256": result.resource_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
