# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Stream the pinned mixed Flash-Next bundle into one native `.ninfer` artifact.

Canonical invocation::

    python -m tools.convert.qwen3_8_flash_next.convert \
      --mixed E:/NInfer/qwen3_8_flash_next/source/mixed \
      --ple E:/NInfer/qwen3_8_flash_next/source/ple \
      --out C:/models/Qwen3.8-Flash-Next/qwen3_8_flash_next_mixed.ninfer
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import struct
import time
from typing import BinaryIO
import sys

# ---------------------------------------------------------------------------------------------
# Front-door bootstrap. THIS MODULE'S OWN DOCSTRING ADVERTISES TWO INVOCATIONS AND ONE OF THEM
# COULD NOT START. MEASURED 2026-09-18 (FN-GATE), from the repository root:
#
#   python3 tools/convert/qwen3_8_flash_next/convert.py --preflight-only
#   ModuleNotFoundError: No module named 'tools'
#
# Running a file by path puts THAT file's directory on sys.path, not the repository root, so the
# `from tools.artifact.container import ...` below cannot resolve -- and the relative
# `from . import inventory, recipe` further down would have failed next with "attempted relative
# import with no known parent package". The `-m` spelling was unaffected and stays unaffected.
# Naming __package__ before the imports is what lets the relative import work; `__package__` is
# empty exactly when the file was run directly, which is also when this is needed.
# ---------------------------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3])
    )
    __package__ = "tools.convert.qwen3_8_flash_next"

import numpy as np

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec as ArtifactResourceSpec,
    TensorObject,
    TensorSpec as ArtifactTensorSpec,
    plan_objects,
)
from tools.artifact.layouts import (
    block_scale_bank_geometry,
    packed_u4_geometry,
    row_scale_geometry,
)
from . import inventory, recipe
from .source import BundlePreflight, SafeTensorDirectory, TensorSlice, validate_bundle, read_safetensor_directory


RECIPE_ID = "qwen3_8_flash_next_mixed-v1"
OUTPUT_BASENAME = "qwen3_8_flash_next_mixed.ninfer"
DEFAULT_PLE_DIR = Path(r"E:\NInfer\qwen3_8_flash_next\source\ple")
DEFAULT_OUTPUT = Path(r"C:\models\Qwen3.8-Flash-Next") / OUTPUT_BASENAME
_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    name: str
    data: bytes


@dataclass(frozen=True, slots=True)
class ObjectPlan:
    specs: tuple[ArtifactResourceSpec | ArtifactTensorSpec, ...]
    objects: tuple[ArtifactObject, ...]

    @property
    def payload_span_bytes(self) -> int:
        last = self.objects[-1]
        return last.offset + last.bytes


class SliceReader(AbstractContextManager["SliceReader"]):
    """Reuse source handles while yielding exact safetensors payload ranges."""

    def __init__(self) -> None:
        self._handles: dict[Path, BinaryIO] = {}

    def _handle(self, path: Path) -> BinaryIO:
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        return handle

    def read_tensor(self, tensor: TensorSlice) -> bytes:
        handle = self._handle(tensor.path)
        handle.seek(tensor.absolute_offset)
        data = handle.read(tensor.bytes)
        if len(data) != tensor.bytes:
            raise ValueError(f"truncated source tensor: {tensor.path}")
        return data

    def iter_tensor(
        self, tensor: TensorSlice, *, chunk_bytes: int = _CHUNK_BYTES
    ) -> Iterable[bytes]:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        handle = self._handle(tensor.path)
        handle.seek(tensor.absolute_offset)
        remaining = tensor.bytes
        while remaining:
            data = handle.read(min(remaining, chunk_bytes))
            if not data:
                raise ValueError(f"truncated source tensor: {tensor.path}")
            remaining -= len(data)
            yield data

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class MixedTensorResolver:
    def __init__(self, mixed_dir: str | Path) -> None:
        self.root = Path(mixed_dir)
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.weight_map: Mapping[str, str] = index["weight_map"]
        self.headers = {
            path.name: read_safetensor_directory(path)
            for path in sorted(self.root.glob("*.safetensors"))
        }

    def tensor(self, name: str) -> TensorSlice:
        shard = self.weight_map[name]
        return self.headers[shard].tensors[name]

    def expert_header(self, layer: int) -> SafeTensorDirectory:
        return self.headers[f"ct-experts-layer{layer:02d}.safetensors"]


def _iter_direct_payload(
    selected: recipe.DirectRecipe,
    resolve: Callable[[str], TensorSlice],
    reader: SliceReader,
) -> Iterable[bytes]:
    if selected.transform in ("copy", "flatten", "concat-rows"):
        for part in selected.sources:
            yield from reader.iter_tensor(resolve(part.name))
        return
    if selected.transform == "conv-channel-major":
        part = selected.sources[0]
        channels, singleton, width = part.shape
        if singleton != 1:
            raise ValueError(f"{selected.object_name}: convolution source is not rank [C,1,W]")
        raw = reader.read_tensor(resolve(part.name))
        words = np.frombuffer(raw, dtype="<u2").reshape(channels, 1, width)
        yield words[:, 0, :].transpose(1, 0).copy().tobytes()
        return
    raise ValueError(f"{selected.object_name}: unknown direct transform {selected.transform}")


def _positive_f32_words(raw: bytes, name: str) -> bytes:
    values = np.frombuffer(raw, dtype="<f4")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name}: FP32 row scales must be finite and positive")
    return raw


def _iter_fp8_payload(
    selected: recipe.Fp8Recipe,
    shape: tuple[int, int],
    resolve: Callable[[str], TensorSlice],
    reader: SliceReader,
) -> Iterable[bytes]:
    geometry = row_scale_geometry(inventory.FP8, shape)
    code_bytes = sum(resolve(matrix.name).bytes for matrix in selected.matrices)
    if code_bytes != geometry.code_plane_bytes:
        raise ValueError(f"{selected.object_name}: fused FP8 code plane has the wrong size")
    for matrix in selected.matrices:
        yield from reader.iter_tensor(resolve(matrix.name))
    padding = geometry.scale_plane_offset - geometry.code_plane_bytes
    if padding:
        yield bytes(padding)
    for matrix in selected.matrices:
        scale_name = matrix.name + "_scale"
        yield _positive_f32_words(reader.read_tensor(resolve(scale_name)), scale_name)


def _iter_ple_payload(
    shape: tuple[int, int],
    tensors: Mapping[str, TensorSlice],
    reader: SliceReader,
) -> Iterable[bytes]:
    geometry = packed_u4_geometry(inventory.PLE_U4, shape)
    codes = tensors["weight_i4"]
    scales = tensors["weight_scale"]
    if codes.bytes != geometry.code_plane_bytes or scales.bytes != geometry.scale_plane_bytes:
        raise ValueError("PLE source planes do not match the registered artifact geometry")
    yield from reader.iter_tensor(codes)
    padding = geometry.scale_plane_offset - geometry.code_plane_bytes
    if padding:
        yield bytes(padding)
    for chunk in reader.iter_tensor(scales):
        values = np.frombuffer(chunk, dtype="<f2")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("PLE group multipliers must be finite and nonnegative")
        yield chunk


def _swizzle_nvfp4_scale_bytes(natural: bytes, n: int, k: int) -> bytes:
    groups = k // 16
    values = np.frombuffer(natural, dtype=np.uint8)
    if values.size != n * groups:
        raise ValueError("NVFP4 natural scale plane has the wrong size")
    if np.any((values & np.uint8(0x80)) != 0) or np.any(values == np.uint8(0x7F)):
        raise ValueError("NVFP4 scales must be nonnegative finite E4M3FN words")
    return (
        values.reshape(n // 128, 4, 32, k // 64, 4)
        .transpose(0, 3, 2, 1, 4)
        .copy()
        .tobytes()
    )


def _expert_name(layer: int, expert: int, projection: str, field: str) -> str:
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{field}"
    )


def _iter_expert_bank_payload(
    selected: recipe.ExpertBankRecipe,
    shape: tuple[int, int, int],
    header: SafeTensorDirectory,
    reader: SliceReader,
) -> Iterable[bytes]:
    geometry = block_scale_bank_geometry(inventory.NVFP4, shape)
    if geometry.experts != 512:
        raise ValueError(f"{selected.object_name}: routed expert count must be 512")

    for expert in range(geometry.experts):
        for projection in selected.projections:
            name = _expert_name(selected.layer, expert, projection, "weight_packed")
            yield from reader.iter_tensor(header.tensors[name])
    padding = geometry.scale_plane_offset - geometry.code_plane_bytes
    if padding:
        yield bytes(padding)

    for expert in range(geometry.experts):
        natural = b"".join(
            reader.read_tensor(
                header.tensors[
                    _expert_name(selected.layer, expert, projection, "weight_scale")
                ]
            )
            for projection in selected.projections
        )
        yield _swizzle_nvfp4_scale_bytes(natural, geometry.n, geometry.k)

    divisors = bytearray()
    for expert in range(geometry.experts):
        words = [
            reader.read_tensor(
                header.tensors[
                    _expert_name(
                        selected.layer, expert, projection, "weight_global_scale"
                    )
                ]
            )
            for projection in selected.projections
        ]
        if any(word != words[0] for word in words[1:]):
            raise ValueError(
                f"{selected.object_name}: fused gate/up expert divisors differ at {expert}"
            )
        _positive_f32_words(words[0], selected.object_name)
        divisors.extend(words[0])
    if len(divisors) != geometry.weight_divisor_bytes:
        raise ValueError(f"{selected.object_name}: expert divisor plane has the wrong size")
    yield divisors


def _ple_header(ple_dir: Path, shard: int) -> SafeTensorDirectory:
    root = ple_dir / "ples_int4" if (ple_dir / "ples_int4").is_dir() else ple_dir
    return read_safetensor_directory(root / f"shard_{shard}.safetensors")


def _load_resources(mixed_dir: Path) -> tuple[ResourcePayload, ...]:
    result = []
    for spec in inventory.RESOURCE_SPECS:
        path = mixed_dir / spec.name.removeprefix("frontend/")
        data = path.read_bytes()
        if not data:
            raise ValueError(f"frontend resource is empty: {path}")
        result.append(ResourcePayload(spec.name, data))
    return tuple(result)


def _object_plan(
    resources: Sequence[ResourcePayload],
) -> ObjectPlan:
    resource_map = {item.name: item.data for item in resources}
    specs: list[ArtifactResourceSpec | ArtifactTensorSpec] = []
    for spec in inventory.OBJECT_SPECS:
        if isinstance(spec, inventory.ResourceSpec):
            specs.append(ArtifactResourceSpec(spec.name, spec.encoding, len(resource_map[spec.name])))
        else:
            specs.append(ArtifactTensorSpec(spec.name, spec.shape, spec.format, spec.layout))
    frozen = tuple(specs)
    return ObjectPlan(frozen, plan_objects(frozen))


def _object_statistics(objects: Sequence[ArtifactObject]) -> dict[str, object]:
    tensors = [item for item in objects if isinstance(item, TensorObject)]
    resources = [item for item in objects if isinstance(item, ResourceObject)]
    object_bytes = sum(item.bytes for item in objects)
    payload_span = objects[-1].offset + objects[-1].bytes
    return {
        "count": len(objects),
        "tensors": len(tensors),
        "resources": len(resources),
        "formats": dict(sorted(Counter(item.format for item in tensors).items())),
        "layouts": dict(sorted(Counter(item.layout for item in tensors).items())),
        "tensor_bytes": sum(item.bytes for item in tensors),
        "resource_bytes": sum(item.bytes for item in resources),
        "object_bytes": object_bytes,
        "payload_span_bytes": payload_span,
        "alignment_bytes": payload_span - object_bytes,
    }


def preflight_conversion(
    mixed_dir: str | Path, ple_dir: str | Path
) -> tuple[
    BundlePreflight,
    recipe.MixedSourceValidation,
    tuple[ResourcePayload, ...],
    ObjectPlan,
]:
    bundle = validate_bundle(mixed_dir, ple_dir)
    mixed_validation = recipe.validate_mixed_source(mixed_dir)
    resources = _load_resources(Path(mixed_dir))
    plan = _object_plan(resources)
    return bundle, mixed_validation, resources, plan


def convert(mixed_dir: str | Path, ple_dir: str | Path, out_path: str | Path) -> Path:
    started = time.perf_counter()
    mixed = Path(mixed_dir)
    ple = Path(ple_dir)
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(f"converter output basename must be {OUTPUT_BASENAME!r}")

    bundle, mixed_validation, resources, plan = preflight_conversion(mixed, ple)
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    required = plan.payload_span_bytes + 1024**3
    if free < required:
        raise OSError(
            f"output volume has {free} free bytes; conversion requires at least {required}"
        )
    print(
        f"preflight complete: {len(plan.objects)} objects, "
        f"{plan.payload_span_bytes / 1024**3:.2f} GiB payload, streaming to {output}",
        flush=True,
    )

    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    resource_map = {item.name: item.data for item in resources}
    resolver = MixedTensorResolver(mixed)
    try:
        with SliceReader() as reader, ArtifactWriter(
            partial,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            plan.specs,
        ) as writer:
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload: bytes | Iterable[bytes] = resource_map[spec.name]
                else:
                    selected = recipe.RECIPES_BY_OBJECT[spec.name]
                    if isinstance(selected, recipe.DirectRecipe):
                        payload = _iter_direct_payload(selected, resolver.tensor, reader)
                    elif isinstance(selected, recipe.Fp8Recipe):
                        payload = _iter_fp8_payload(selected, spec.shape, resolver.tensor, reader)
                    elif isinstance(selected, recipe.ExpertBankRecipe):
                        payload = _iter_expert_bank_payload(
                            selected, spec.shape, resolver.expert_header(selected.layer), reader
                        )
                    elif isinstance(selected, recipe.PleRecipe):
                        payload = _iter_ple_payload(
                            spec.shape, _ple_header(ple, selected.shard).tensors, reader
                        )
                    else:
                        raise TypeError(f"unsupported recipe: {type(selected).__name__}")
                writer.write(spec.name, payload)
                print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - started
    report = {
        "identity": {"model_id": inventory.MODEL_ID, "weights_id": inventory.WEIGHTS_ID},
        "target_key": inventory.TARGET_KEY,
        "recipe_id": RECIPE_ID,
        "source": {
            "mixed_path": str(mixed.resolve()),
            "mixed_revision": bundle.resource_sha256 and "a4e813ed3cfbbcc61e2929699eccb864a4dfa843",
            "ple_path": str(bundle.ple_dir.resolve()),
            "ple_revision": "da8b39586016d8325ac619be28ad77d6296625ec",
        },
        "source_preflight": {
            "index_tensors": mixed_validation.index_tensors,
            "consumed_tensors": mixed_validation.consumed_tensors,
            "replaced_ple_tensors": mixed_validation.replaced_ple_tensors,
            "expert_tensors": bundle.expert_tensors,
            "ple_shards": bundle.ple_shards,
            "ple_rows": bundle.ple_rows,
            "resource_sha256": dict(bundle.resource_sha256),
        },
        "objects": _object_statistics(plan.objects),
        "elapsed_seconds": elapsed,
        "artifact": {"path": str(output.resolve()), "bytes": output.stat().st_size},
    }
    report_path = Path(str(output) + ".conversion.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"complete: {output.stat().st_size} bytes in {elapsed:.1f}s", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", type=Path, default=recipe.DEFAULT_MIXED_DIR)
    parser.add_argument("--ple", type=Path, default=DEFAULT_PLE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        bundle, mixed_validation, _, plan = preflight_conversion(args.mixed, args.ple)
        print(
            json.dumps(
                {
                    "mixed": str(bundle.mixed_dir),
                    "ple": str(bundle.ple_dir),
                    "index_tensors": mixed_validation.index_tensors,
                    "objects": len(plan.objects),
                    "payload_bytes": plan.payload_span_bytes,
                },
                indent=2,
            )
        )
        return
    convert(args.mixed, args.ple, args.out)


if __name__ == "__main__":
    main()
