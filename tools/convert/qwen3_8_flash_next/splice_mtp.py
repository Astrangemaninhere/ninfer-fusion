# Derived from igorls/ninfer @ 5e4a66d (Apache-2.0), modified for this tree (ninfer-fusion).
"""Splice NVFP4 quantized MTP expert banks into a Qwen3.8-Flash-Next artifact.

Produces an exact, structurally valid .ninfer v2 container where
mtp/layer/mlp/experts/gate_up and mtp/layer/mlp/experts/down are replaced
with NVFP4 blockscale layouts, shrinking the artifact on disk by ~3.37 GiB
without requiring a full 105 GiB re-convert from source safetensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Iterator

from tools.artifact.container import (
    Artifact,
    ArtifactWriter,
    ResourceObject,
    ResourceSpec,
    TensorObject,
    TensorSpec,
)
from tools.artifact.layouts import block_scale_bank_geometry

DEFAULT_SRC = Path(r"C:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_mixed.ninfer")
DEFAULT_DST = Path(r"E:\models\Qwen3.8-Flash-Next\qwen3_8_flash_next_nvfp4_mtp.ninfer")
DEFAULT_GATE_UP_BIN = Path(r"E:\models\Qwen3.8-Flash-Next\mtp_gate_up_nvfp4.bin")
DEFAULT_DOWN_BIN = Path(r"E:\models\Qwen3.8-Flash-Next\mtp_down_nvfp4.bin")

CHUNK_SIZE = 64 * 1024 * 1024  # 64 MiB streaming buffer


def _chunk_file_range(
    f: BinaryIO, offset: int, length: int, chunk_size: int = CHUNK_SIZE
) -> Iterator[bytes]:
    f.seek(offset)
    remaining = length
    while remaining > 0:
        to_read = min(chunk_size, remaining)
        chunk = f.read(to_read)
        if not chunk:
            break
        yield chunk
        remaining -= len(chunk)


def _chunk_file(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def splice_artifact(
    src_path: Path,
    dst_path: Path,
    gate_up_bin: Path,
    down_bin: Path,
) -> None:
    started = time.perf_counter()
    print(f"Reading source artifact: {src_path}", flush=True)

    expected_gate_up_bytes = block_scale_bank_geometry("NVFP4", (512, 1280, 2560)).payload_bytes
    expected_down_bytes = block_scale_bank_geometry("NVFP4", (512, 2560, 640)).payload_bytes

    if not gate_up_bin.is_file():
        raise FileNotFoundError(f"Quantized gate_up binary not found: {gate_up_bin}")
    if not down_bin.is_file():
        raise FileNotFoundError(f"Quantized down binary not found: {down_bin}")

    actual_gate_up_bytes = gate_up_bin.stat().st_size
    actual_down_bytes = down_bin.stat().st_size
    if actual_gate_up_bytes != expected_gate_up_bytes:
        raise ValueError(
            f"gate_up size mismatch: expected {expected_gate_up_bytes}, got {actual_gate_up_bytes}"
        )
    if actual_down_bytes != expected_down_bytes:
        raise ValueError(
            f"down size mismatch: expected {expected_down_bytes}, got {actual_down_bytes}"
        )

    with Artifact.open(src_path) as src, src_path.open("rb") as src_file:
        new_specs: list[TensorSpec | ResourceSpec] = []
        for obj in src.objects:
            if obj.name == "mtp/layer/mlp/experts/gate_up":
                new_specs.append(
                    TensorSpec(
                        name=obj.name,
                        shape=(512, 1280, 2560),
                        format="NVFP4",
                        layout="expert-blockscale-k16-m128x4-v1",
                    )
                )
            elif obj.name == "mtp/layer/mlp/experts/down":
                new_specs.append(
                    TensorSpec(
                        name=obj.name,
                        shape=(512, 2560, 640),
                        format="NVFP4",
                        layout="expert-blockscale-k16-m128x4-v1",
                    )
                )
            elif isinstance(obj, TensorObject):
                new_specs.append(
                    TensorSpec(
                        name=obj.name,
                        shape=obj.shape,
                        format=obj.format,
                        layout=obj.layout,
                    )
                )
            elif isinstance(obj, ResourceObject):
                new_specs.append(
                    ResourceSpec(
                        name=obj.name,
                        encoding=obj.encoding,
                        bytes=obj.bytes,
                    )
                )
            else:
                raise TypeError(f"Unknown object type: {type(obj).__name__}")

        print(f"Opening destination writer: {dst_path}", flush=True)
        with ArtifactWriter(dst_path, src.identity, new_specs) as writer:
            total_objs = len(src.objects)
            bytes_streamed = 0
            last_report = time.perf_counter()

            for idx, obj in enumerate(src.objects):
                if obj.name == "mtp/layer/mlp/experts/gate_up":
                    print(f"[{idx+1}/{total_objs}] Splicing NVFP4 gate_up ({actual_gate_up_bytes} B)...", flush=True)
                    writer.write(obj.name, _chunk_file(gate_up_bin))
                    bytes_streamed += actual_gate_up_bytes
                elif obj.name == "mtp/layer/mlp/experts/down":
                    print(f"[{idx+1}/{total_objs}] Splicing NVFP4 down ({actual_down_bytes} B)...", flush=True)
                    writer.write(obj.name, _chunk_file(down_bin))
                    bytes_streamed += actual_down_bytes
                else:
                    abs_offset = src.payload_offset + obj.offset
                    writer.write(obj.name, _chunk_file_range(src_file, abs_offset, obj.bytes))
                    bytes_streamed += obj.bytes

                now = time.perf_counter()
                if now - last_report >= 5.0 or idx == total_objs - 1:
                    rate_mb_s = (bytes_streamed / 1024**2) / (now - started)
                    print(
                        f"Progress: [{idx+1}/{total_objs}] objects written | "
                        f"{bytes_streamed / 1024**3:.2f} GiB streamed ({rate_mb_s:.1f} MiB/s)",
                        flush=True,
                    )
                    last_report = now

    elapsed = time.perf_counter() - started
    final_size = dst_path.stat().st_size
    diff_gib = (src_path.stat().st_size - final_size) / 1024**3
    print(
        f"Splice complete in {elapsed:.1f}s!\n"
        f"Output file: {dst_path} ({final_size} bytes, {final_size/1024**3:.2f} GiB)\n"
        f"Disk savings: {diff_gib:.3f} GiB",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source .ninfer artifact")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Destination .ninfer artifact")
    parser.add_argument(
        "--gate-up-bin",
        type=Path,
        default=DEFAULT_GATE_UP_BIN,
        help="Path to quantized gate_up NVFP4 payload",
    )
    parser.add_argument(
        "--down-bin",
        type=Path,
        default=DEFAULT_DOWN_BIN,
        help="Path to quantized down NVFP4 payload",
    )
    args = parser.parse_args()

    splice_artifact(args.src, args.dst, args.gate_up_bin, args.down_bin)


if __name__ == "__main__":
    main()
