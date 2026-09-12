"""Byte-exact assembly of the 12 ``mtp/*`` objects of the qwen3.8-27B NVFP4 artifact.

Method (deliberately *not* the whole-matrix ``quantize_matrix`` route):

* the low-level persistent-layout primitives ``row_split_geometry`` and
  ``assemble_row_planes`` from ``tools/artifact/layouts.py`` are reused directly;
* the group statistics are reproduced from scratch, host-side, with the exact
  canonical rule of ``tools/convert/common/quantize.py:_canonical_scale_words``
  (``float16(float32(float64(max_abs) / qmax))`` with the ``2**-24`` subnormal
  floor, then ``reciprocal = float32(1.0 / float64(scale))``);
* the int8 codes are ``clamp(round(x * reciprocal), -127, 127)``;
* rows are consumed in bounded row segments so that a 34816x5120 matrix never
  exists as a single float32 buffer.

Public interface (identical to the sibling implementation):

    build_mtp_objects(source_file: str | Path, device="cpu") -> dict[str, bytes]
"""

from __future__ import annotations

import json
import mmap
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

# --------------------------------------------------------------------------
# repository bootstrap (the repo is imported, never modified)
# --------------------------------------------------------------------------

_REPO_CANDIDATES = (
    Path("/home/user/ninfer-fusion"),
    Path(__file__).resolve().parents[1],
)


def _bootstrap_repo() -> Path:
    try:
        import tools.artifact.layouts  # noqa: F401
    except ImportError:
        pass
    else:
        import tools.artifact.layouts as _l

        return Path(_l.__file__).resolve().parent.parent.parent
    for candidate in _REPO_CANDIDATES:
        if (candidate / "tools" / "artifact" / "layouts.py").is_file():
            sys.path.insert(0, str(candidate))
            return candidate
    raise ImportError("cannot locate the ninfer-fusion repository root")


REPO_ROOT = _bootstrap_repo()

from tools.artifact.layouts import (  # noqa: E402
    RowPlanes,
    assemble_row_planes,
    encode_direct,
    row_split_geometry,
)
from tools.artifact.numeric import get_format  # noqa: E402


W8 = "W8G32_F16S"
BF16 = "BF16"

#: ``_FP16_MIN_SUBNORMAL`` from the repository's grouped-quantization module.
_FP16_MIN_SUBNORMAL = 2.0**-24

#: Target size of the temporary buffers used for one row segment.
_SEGMENT_TARGET_BYTES = 32 * 1024 * 1024

#: Row segment granularity, kept a multiple of the 128-row K alignment.
_SEGMENT_ROW_QUANTUM = 16


# --------------------------------------------------------------------------
# source access: mmap-backed, zero-copy, row addressable
# --------------------------------------------------------------------------


class SafeTensorSource:
    """Minimal read-only safetensors reader with row-granular zero-copy reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            header_bytes = int.from_bytes(self._file.read(8), "little")
            header = json.loads(self._file.read(header_bytes).decode("utf-8"))
            self._header = {
                key: value for key, value in header.items() if key != "__metadata__"
            }
            self._data_start = 8 + header_bytes
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self._file.close()
            raise

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._header)

    def has(self, name: str) -> bool:
        return name in self._header

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._header[name]["shape"])

    def dtype(self, name: str) -> str:
        return self._header[name]["dtype"]

    def raw(self, name: str) -> memoryview:
        begin, end = self._header[name]["data_offsets"]
        assert self._mapping is not None
        return memoryview(self._mapping)[self._data_start + begin : self._data_start + end]

    def rows_float32(self, name: str, begin: int, count: int) -> np.ndarray:
        """Return rows ``[begin, begin+count)`` widened to float32.

        The widening is exact for the BF16 payload of this checkpoint
        (``bfloat16 -> float32`` is a pure exponent re-bias, i.e. a ``<< 16``),
        which is what the repository oracle performs as ``.to(torch.float32)``.
        """

        if self.dtype(name) != "BF16":
            raise TypeError(f"{name}: expected BF16 source, got {self.dtype(name)}")
        rows, columns = self.shape(name)
        if begin < 0 or count <= 0 or begin + count > rows:
            raise IndexError(f"{name}: row range [{begin},{begin + count}) outside {rows}")
        words = np.frombuffer(
            self.raw(name),
            dtype=np.uint16,
            count=count * columns,
            offset=begin * columns * 2,
        )
        return (words.astype(np.int32) << 16).view(np.float32).reshape(count, columns)

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "SafeTensorSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# canonical grouped statistics (host oracle rule, reproduced verbatim)
# --------------------------------------------------------------------------


def canonical_scale_words(
    max_abs: np.ndarray, qmax: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(float16 scales, float32 reciprocals)`` for BF16 group maxima.

    Reproduces ``tools/convert/common/quantize.py:_canonical_scale_words``
    exactly: the host divides in binary64, rounds explicitly through binary32 and
    then binary16, floors positive underflow at the smallest fp16 subnormal, and
    recomputes the reciprocal as ``float32(1.0 / float64(scale))``.
    """

    host_max = np.asarray(max_abs, dtype=np.float32)
    if not np.isfinite(host_max).all():
        raise ValueError("grouped quantization source contains NaN or infinity")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_scale = (host_max.astype(np.float64) / float(qmax)).astype(np.float32)
        scale = raw_scale.astype(np.float16)
    underflow = (scale == 0) & (host_max > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    if np.any((host_max > 0) & (~np.isfinite(scale) | (scale <= 0))):
        raise ValueError("grouped quantization scale is not finite and positive")

    reciprocal = np.zeros(host_max.shape, dtype=np.float32)
    positive = scale > 0
    reciprocal[positive] = (1.0 / scale[positive].astype(np.float64)).astype(np.float32)
    return scale, reciprocal


def segment_geometry(k: int, target_bytes: int = _SEGMENT_TARGET_BYTES) -> int:
    """Pick a row-segment size that bounds the transient buffers."""

    per_row = 12 * k + 64
    rows = max(1, target_bytes // per_row)
    rows = max(_SEGMENT_ROW_QUANTUM, rows - rows % _SEGMENT_ROW_QUANTUM)
    return rows


def quantize_segment(
    logical: np.ndarray,
    spec,
    k_pad: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one row segment into ``(base bytes, float16 scales)``."""

    rows = logical.shape[0]
    # Always own the buffer. With k_pad == k the incoming block can alias the
    # caller's array (np.ascontiguousarray is a no-op for float32 C order), and the
    # in-place quantization below would then rewrite the caller's data - which
    # silently corrupts any later use of that array. A differential test against
    # the repository quantizer caught exactly that, so the copy is unconditional.
    block = np.zeros((rows, k_pad), dtype=np.float32)
    block[:, : logical.shape[1]] = logical
    grouped = block.reshape(rows, k_pad // spec.group_size, spec.group_size)
    max_abs = np.abs(grouped).max(axis=2)
    scales, reciprocal = canonical_scale_words(max_abs, spec.qmax)
    # In place: `block` is owned by this segment.
    grouped *= reciprocal[:, :, None]
    np.round(grouped, out=grouped)
    np.clip(grouped, spec.qmin, spec.qmax, out=grouped)
    base = np.ascontiguousarray(grouped.astype(np.int8)).reshape(-1).view(np.uint8)
    return base, scales


# --------------------------------------------------------------------------
# object plan
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RowRun:
    """``rows`` consecutive source rows of ``source_key`` starting at ``source_row``."""

    source_key: str
    source_row: int
    rows: int


@dataclass(frozen=True, slots=True)
class VectorObject:
    object_name: str
    source_key: str


@dataclass(frozen=True, slots=True)
class MatrixObject:
    object_name: str
    shape: tuple[int, int]
    runs: tuple[RowRun, ...]


_L = "mtp.layers.0."
_QP = _L + "self_attn.q_proj.weight"
_KP = _L + "self_attn.k_proj.weight"
_VP = _L + "self_attn.v_proj.weight"
_GP = _L + "mlp.gate_proj.weight"
_UP = _L + "mlp.up_proj.weight"

HEADS = 24
HEAD_ROWS = 512
PART_ROWS = 256


def _q_part(key: str, gate: bool) -> tuple[RowRun, ...]:
    """Head-interleaved q projection rows: per head 512 rows, first/second 256."""

    offset = PART_ROWS if gate else 0
    return tuple(
        RowRun(key, head * HEAD_ROWS + offset, PART_ROWS) for head in range(HEADS)
    )


def _whole(key: str, rows: int) -> tuple[RowRun, ...]:
    return (RowRun(key, 0, rows),)


#: Named candidate row orders for the fused attention matrix, scored against the
#: oracle by ``scratch/t2/_judge.py``.  The confirmed winner is selected below.
QKGV_CANDIDATES: dict[str, tuple[RowRun, ...]] = {
    "q_norm|k|q_gate|v": _q_part(_QP, False) + _whole(_KP, 1024) + _q_part(_QP, True)
    + _whole(_VP, 1024),
    "q|k|v": _whole(_QP, 12288) + _whole(_KP, 1024) + _whole(_VP, 1024),
    "q_gate|k|q_norm|v": _q_part(_QP, True) + _whole(_KP, 1024) + _q_part(_QP, False)
    + _whole(_VP, 1024),
    "k|v|q_norm|q_gate": _whole(_KP, 1024) + _whole(_VP, 1024) + _q_part(_QP, False)
    + _q_part(_QP, True),
    "q_norm|k|v|q_gate": _q_part(_QP, False) + _whole(_KP, 1024) + _whole(_VP, 1024)
    + _q_part(_QP, True),
    "k|q_norm|q_gate|v": _whole(_KP, 1024) + _q_part(_QP, False) + _q_part(_QP, True)
    + _whole(_VP, 1024),
}

#: Named candidate row orders for the fused MLP matrix.
GATE_UP_CANDIDATES: dict[str, tuple[RowRun, ...]] = {
    "gate|up": _whole(_GP, 17408) + _whole(_UP, 17408),
    "up|gate": _whole(_UP, 17408) + _whole(_GP, 17408),
}

#: Bit-exact winners, established with the oracle as a byte-level judge
#: (see REPORT.md for the candidate hit counts).
QKGV_ORDER = "q_norm|k|q_gate|v"
GATE_UP_ORDER = "gate|up"


MTP_OBJECTS: tuple[VectorObject | MatrixObject, ...] = (
    MatrixObject("mtp/input_projection", (5120, 10240), _whole("mtp.fc.weight", 5120)),
    VectorObject("mtp/embedding_norm", "mtp.pre_fc_norm_embedding.weight"),
    VectorObject("mtp/hidden_norm", "mtp.pre_fc_norm_hidden.weight"),
    VectorObject("mtp/layer/input_norm", _L + "input_layernorm.weight"),
    MatrixObject(
        "mtp/layer/attention/query_key_gate_value",
        (14336, 5120),
        QKGV_CANDIDATES[QKGV_ORDER],
    ),
    VectorObject("mtp/layer/attention/query_norm", _L + "self_attn.q_norm.weight"),
    VectorObject("mtp/layer/attention/key_norm", _L + "self_attn.k_norm.weight"),
    MatrixObject(
        "mtp/layer/attention/output",
        (5120, 6144),
        _whole(_L + "self_attn.o_proj.weight", 5120),
    ),
    VectorObject("mtp/layer/post_attention_norm", _L + "post_attention_layernorm.weight"),
    MatrixObject(
        "mtp/layer/mlp/gate_up", (34816, 5120), GATE_UP_CANDIDATES[GATE_UP_ORDER]
    ),
    MatrixObject(
        "mtp/layer/mlp/down", (5120, 17408), _whole(_L + "mlp.down_proj.weight", 5120)
    ),
    VectorObject("mtp/final_norm", "mtp.norm.weight"),
)


def iter_row_runs(runs: Sequence[RowRun], block_rows: int) -> Iterator[tuple[int, RowRun]]:
    """Split the ordered run list into ``(output_row, RowRun)`` chunks of bounded size."""

    output_row = 0
    for run in runs:
        offset = 0
        while offset < run.rows:
            count = min(block_rows, run.rows - offset)
            yield output_row, RowRun(run.source_key, run.source_row + offset, count)
            output_row += count
            offset += count


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------


def encode_vector(reader: SafeTensorSource, spec: VectorObject) -> bytes:
    import torch

    words = np.frombuffer(reader.raw(spec.source_key), dtype=np.uint16)
    tensor = torch.from_numpy(words.copy()).view(torch.bfloat16)
    return encode_direct(tensor, BF16)


def encode_matrix(
    reader: SafeTensorSource,
    spec: MatrixObject,
    *,
    block_rows: int | None = None,
) -> bytes:
    """Stream a bounded number of rows at a time into a preallocated payload.

    Each segment is quantized independently and encoded with the repository's
    ``assemble_row_planes``; because ``base_row_bytes``/``scale_row_bytes`` are
    independent of the row count, the segment planes can be written straight into
    the destination plane offsets of the full geometry.
    """

    n, k = spec.shape
    numeric = get_format(W8)
    geometry = row_split_geometry(numeric, spec.shape)
    if geometry.k_pad % numeric.group_size:
        raise AssertionError("K padding is not group aligned")

    payload = bytearray(geometry.payload_bytes)
    view = memoryview(payload)
    base_plane = view[: geometry.base_bytes]
    scale_plane = view[
        geometry.scale_offset : geometry.scale_offset + geometry.scale_bytes
    ]

    if block_rows is None:
        block_rows = segment_geometry(k)

    for output_row, run in iter_row_runs(spec.runs, block_rows):
        logical = reader.rows_float32(run.source_key, run.source_row, run.rows)
        base, scales = quantize_segment(logical, numeric, geometry.k_pad)
        block = assemble_row_planes(
            RowPlanes(
                base,
                np.empty(0, dtype=np.uint8),
                scales.reshape(-1).view(np.uint8),
                run.rows,
            ),
            numeric,
            k,
        )
        block_view = memoryview(block).cast("B")
        block_geometry = row_split_geometry(numeric, (run.rows, k))
        begin = output_row * geometry.base_row_bytes
        base_plane[begin : begin + block_geometry.base_bytes] = block_view[
            : block_geometry.base_bytes
        ]
        scale_begin = output_row * geometry.scale_row_bytes
        scale_plane[
            scale_begin : scale_begin + block_geometry.scale_bytes
        ] = block_view[
            block_geometry.scale_offset : block_geometry.scale_offset
            + block_geometry.scale_bytes
        ]
        del block, block_view, base, scales, logical

    result = bytes(payload)
    return result


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def source_keys(spec: VectorObject | MatrixObject) -> tuple[str, ...]:
    if isinstance(spec, VectorObject):
        return (spec.source_key,)
    return tuple(dict.fromkeys(run.source_key for run in spec.runs))


def build_mtp_objects(source_file: str | Path, device: str = "cpu") -> dict[str, bytes]:
    """Rebuild the 12 ``mtp/*`` payloads byte-for-byte from the BF16 MTP source.

    ``device`` is accepted for interface parity only.  The registered group-scale
    rule is defined on the host (binary64 division explicitly rounded through
    binary32 and then binary16), i.e. it is a host-side oracle by construction,
    so the arithmetic is pinned to the host whatever the request.  A non-CPU
    value is therefore tolerated but has no effect on the produced bytes.
    """

    del device  # accepted for parity; the arithmetic is host-authored (see above)
    source = Path(source_file)
    with SafeTensorSource(source) as reader:
        missing = [
            key
            for spec in MTP_OBJECTS
            for key in source_keys(spec)
            if not reader.has(key)
        ]
        if missing:
            raise KeyError(f"source is missing tensors: {missing}")
        objects: dict[str, bytes] = {}
        for spec in MTP_OBJECTS:
            if isinstance(spec, VectorObject):
                objects[spec.object_name] = encode_vector(reader, spec)
            else:
                objects[spec.object_name] = encode_matrix(reader, spec)
        return objects


__all__ = [
    "GATE_UP_CANDIDATES",
    "MTP_OBJECTS",
    "QKGV_CANDIDATES",
    "SafeTensorSource",
    "build_mtp_objects",
    "canonical_scale_words",
    "encode_matrix",
    "segment_geometry",
    "source_keys",
]
