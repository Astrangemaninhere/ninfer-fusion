# -*- coding: utf-8 -*-
"""Logical source reader for a ``compressed-tensors`` NVFP4 HF checkpoint.

This is the sibling of :mod:`tools.convert.dequant.modelopt` for the *other*
NVFP4 field vocabulary.  The two checkpoints store the same four quantities::

    this source (compressed-tensors)      modelopt reader
    --------------------------------      ------------------------------
    <base>.weight_packed        U8        <base>.weight        U8
    <base>.weight_scale         F8_E4M3   <base>.weight_scale   F8_E4M3
    <base>.weight_global_scale  F32       <base>.weight_scale_2 F32   (see below)
    <base>.input_global_scale   F32       <base>.input_scale    F32   (see below)

⚠ **Two of the four are VALUE maps, not name maps.**  The global scales are the
RECIPROCAL of the modelopt pair, and both facts are read off the vendored
upstream implementation rather than remembered:

* ``compressed_tensors/entrypoints/convert/converters/modelopt_nvfp4.py:46``
  states the map in its own docstring -- ``- 1 / weight_scale_2 ->
  weight_global_scale`` -- and ``:62``/``:74`` perform it
  (``tensors[f"{module_name}.input_global_scale"] = 1 / tensors[name]``);
* ``compressed_tensors/quantization/lifecycle/forward_helpers.py:246-256``
  (``_dequantize``) computes ``scale = scale / global_scale`` and then
  ``x_q * scale``, i.e. the global scale is a DIVISOR of the stored block scale.

Consequently the engine's ``weight_divisor`` -- the one FP32 word
``tools.artifact.layouts.encode_nvfp4`` stores, and the quantity the engine
divides by -- is ``fp32(weight_global_scale)`` **taken as stored**, and *not* its
reciprocal.  ``ModelOptSource.nvfp4_words`` returns ``fp32(1 / weight_scale_2)``
for its vocabulary, and for this vocabulary that equals the stored global scale;
inverting it here would be wrong by a factor of ``global_scale**2``.  That
identity is what :meth:`CompressedTensorsSource.divisor` computes, and
:data:`DIVISOR_IS_THE_STORED_WORD` records the decision in one name.

Measured storage contract of this source
----------------------------------------

==========================================  ============  ===========================
key                                         stored dtype  stored shape
==========================================  ============  ===========================
``<base>.weight_packed``                    ``U8``        ``(N, K/2)``
``<base>.weight_scale``                     ``F8_E4M3``   ``(N, K/16)`` natural
``<base>.weight_global_scale``              ``F32``       ``(1,)`` scalar divisor
``<base>.input_global_scale``               ``F32``       ``(1,)`` scalar multiplier
==========================================  ============  ===========================

* the scale plane is **natural (unswizzled) row order**: upstream's compressor
  stores it with ``_compress_scale`` = ``scale.to(scale_dtype)`` and no
  permutation at all (``compressors/nvfp4/base.py:53-56``), and its
  modelopt converter passes ``weight_scale`` through unchanged (``:70``).  The
  m128x4 swizzle is applied by the *kernel consumer*, so it is
  ``layouts.encode_nvfp4``'s job -- which is exactly what it does;
* element ``2j`` lives in the low nibble of packed byte ``j`` and element
  ``2j+1`` in its high nibble (``compressors/nvfp4/helpers.py:66``:
  ``indices[:, 0] | (indices[:, 1] << 4)``);
* every other key (norms, ``layer_scalar``, embeddings, vision / audio towers)
  is a plain BF16 or FP32 tensor and is returned as stored.

Nothing is loaded up front: construction reads the index (or the shard headers)
only, and every payload access is a row-range slice taken from the shard handle,
so one block of ``chunk_rows`` rows is the largest temporary the dequantizing
path allocates.

Canonical use::

    from tools.convert.dequant.compressed_tensors import CompressedTensorsSource

    with CompressedTensorsSource("/models/gemma4-31B") as source:
        for base in source.quantized_keys():
            packed, scales, divisor = source.nvfp4_words(base)
            payload = source.encode_payload(base)          # layouts.encode_nvfp4

Standalone inspection (no artifact, no GPU)::

    python3 -m tools.convert.dequant.compressed_tensors --model /models/gemma4-31B
    python3 -m tools.convert.dequant.compressed_tensors --model /models/gemma4-31B \\
        --self-check --encode model.language_model.layers.0.mlp.gate_proj
"""

from __future__ import annotations

import argparse
import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


__all__ = [
    "DEFAULT_CHUNK_ROWS",
    "CompressedTensorsSource",
    "FIELD_INPUT_GLOBAL_SCALE",
    "FIELD_PACKED",
    "FIELD_SCALE",
    "FIELD_WEIGHT_GLOBAL_SCALE",
    "FIELDS",
    "MODELOPT_EQUIVALENT",
    "REFUSALS",
]


DEFAULT_CHUNK_ROWS = 1024
"""Logical rows decoded per block; the reader's only peak-memory knob."""

INDEX_NAME = "model.safetensors.index.json"
SINGLE_SHARD_NAME = "model.safetensors"
SHARD_GLOB = "*.safetensors"

GROUP = 16
"""Logical elements sharing one E4M3FN scale word along K."""

FIELD_PACKED = "weight_packed"
FIELD_SCALE = "weight_scale"
FIELD_WEIGHT_GLOBAL_SCALE = "weight_global_scale"
FIELD_INPUT_GLOBAL_SCALE = "input_global_scale"
FIELDS: tuple[str, ...] = (
    FIELD_PACKED,
    FIELD_SCALE,
    FIELD_WEIGHT_GLOBAL_SCALE,
    FIELD_INPUT_GLOBAL_SCALE,
)

#: Name map onto :mod:`tools.convert.dequant.modelopt`'s vocabulary.  The first
#: two are identity renames; the last two are reciprocal VALUE maps -- see the
#: module docstring.  ``MODELOPT_RECIPROCAL`` names which ones.
MODELOPT_EQUIVALENT: Mapping[str, str] = MappingProxyType(
    {
        FIELD_PACKED: "weight",
        FIELD_SCALE: "weight_scale",
        FIELD_WEIGHT_GLOBAL_SCALE: "weight_scale_2",
        FIELD_INPUT_GLOBAL_SCALE: "input_scale",
    }
)
MODELOPT_RECIPROCAL: tuple[str, ...] = (FIELD_WEIGHT_GLOBAL_SCALE, FIELD_INPUT_GLOBAL_SCALE)

COMPILED_MARKER = "ctspell.compressed-tensors.v1"
"""A string literal that is (a) in the source, (b) in the compiled code object and
(c) printed by :meth:`CompressedTensorsSource.__repr__`.

R43/R52 want a marker that survives compilation and that a probe can count BY
OCCURRENCE, so a build gate can prove it compiled *this* file rather than a
neighbour of the same name.  A module-level string constant lands in the code
object's ``co_consts`` and in the ``.pyc`` string table, and this one is also
reachable at run time, so the probe needs no private knowledge of the compiler.
"""

DIVISOR_IS_THE_STORED_WORD = True
"""``weight_divisor = fp32(weight_global_scale)``; never its reciprocal."""

# --------------------------------------------------------------------------- #
# Refusal reasons, each a name a probe can match literally (R43/R52).
# --------------------------------------------------------------------------- #
REFUSALS: Mapping[str, str] = MappingProxyType(
    {
        "incomplete_group": "NVFP4 group does not carry all four stored fields",
        "packed_not_uint8": "stored packed codes are not the U8 word plane",
        "scale_not_e4m3fn": "stored scale plane is not F8_E4M3 words",
        "global_scale_not_fp32": "global scale must be one F32 word",
        "global_scale_not_positive": "global scale must be finite and positive",
        "scale_columns": "scale columns must be exactly K/16",
        "group_size": "K must be a multiple of the 16-element group",
        "scale_codes": "NVFP4 scale words must be nonnegative finite E4M3FN codes",
        "packed_columns": "packed columns must be exactly K/2",
        "unknown_key": "is not a key of this checkpoint",
        "not_quantized": "is not an NVFP4-quantized weight of this checkpoint",
        "dtype_unsupported": "unsupported stored dtype",
    }
)

_TORCH_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E4M3FN": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }
)

#: The seven E2M1 magnitudes, indexed by the magnitude field (bit 3 is the sign).
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

_PACKED_DTYPES = ("U8", "I8")


def _e2m1_code_bits() -> np.ndarray:
    """Binary32 word bits of every four-bit E2M1 code (bit 3 is the sign)."""

    values = [
        np.float32(
            np.copysign(_E2M1_MAGNITUDES[code & 0x7], -1.0 if code & 0x8 else 1.0)
        )
        for code in range(16)
    ]
    return np.array(values, dtype=np.float32).view(np.uint32)


_E2M1_BITS = _e2m1_code_bits()


def _e2m1_pair_words() -> np.ndarray:
    """``(256,)`` table: one packed byte -> the two binary32 words it encodes.

    Element ``2j`` is the low nibble and ``2j+1`` the high nibble, which is the
    packing upstream's ``pack_fp4_to_uint8`` writes.  On a little-endian host a
    single gather followed by a ``float32`` view yields logical element order
    directly.
    """

    codes = np.arange(256, dtype=np.uint32)
    low = _E2M1_BITS[codes & 0x0F].astype(np.uint64)
    high = _E2M1_BITS[codes >> 4].astype(np.uint64)
    return (low | (high << np.uint64(32))).astype(np.uint64)


_E2M1_PAIR_WORDS = _e2m1_pair_words()

_SUBNORMAL_BITS = np.array(
    [np.float32(fraction * 2.0**-9) for fraction in range(8)], dtype=np.float32
).view(np.uint32)
"""Binary32 bits of ``fraction * 2**-9``, the E4M3FN subnormal range."""


def _e4m3fn_values(words: np.ndarray) -> np.ndarray:
    """Expand E4M3FN words to their exact binary32 values, by integer bits.

    A normal E4M3FN word maps into binary32 without rounding -- exponent field
    ``exponent + 120`` with ``fraction`` in the top three mantissa bits -- so the
    whole expansion is one vectorized integer pipeline; only the eight subnormal
    codes need a table.  ``0x7F`` / ``0xFF`` are not represented faithfully and
    callers screen them out first (:func:`_require_scale_words`).
    """

    word = np.ascontiguousarray(words, dtype=np.uint32)
    sign = (word & 0x80) << 24
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    normal = ((exponent + 120) << 23) | (fraction << 20)
    bits = np.where(exponent == 0, _SUBNORMAL_BITS[fraction], normal)
    return (bits.astype(np.uint32) | sign).view(np.float32)


def _require_scale_words(words: torch.Tensor, label: str) -> None:
    """Reject E4M3FN scale words that are negative, non-finite, or ``0x7F``."""

    if bool((((words & 0x80) != 0) | (words == 0x7F)).any()):
        raise ValueError("%s: %s" % (label, REFUSALS["scale_codes"]))


def _flat_uint8(tensor: torch.Tensor, rows: int, columns: int) -> np.ndarray:
    """Host ``(rows, columns)`` uint8 view of one raw-word block."""

    return tensor.reshape(rows, columns).contiguous().numpy()


def _dequantize_block(packed: np.ndarray, scales: np.ndarray, divisor: float) -> np.ndarray:
    """Vectorized ``E2M1(nibble) * E4M3FN(scale) / divisor`` in binary32.

    The division is the one ``forward_helpers.py:246-256`` documents: upstream
    divides the stored block scale by the global scale, so the engine's single
    trailing divisor is the global scale itself.
    """

    rows, half = packed.shape
    column = half * 2
    groups = scales.shape[1]
    if column != groups * GROUP:
        raise ValueError("packed codes and scale words disagree about the K axis")
    values = _E2M1_PAIR_WORDS[np.ascontiguousarray(packed)].view(np.float32)
    scale_values = _e4m3fn_values(scales)
    values = values.reshape(rows, groups, GROUP) * scale_values[:, :, None]
    divisor32 = np.float32(divisor)
    if not np.isfinite(divisor32) or divisor32 <= 0.0:
        raise ValueError(REFUSALS["global_scale_not_positive"])
    values /= divisor32
    return values.reshape(rows, column)


def _fp32_scalar(tensor: torch.Tensor, label: str) -> float:
    """Exact Python float of a stored FP32 scalar (a binary32 round-trip)."""

    if tensor.numel() != 1:
        raise ValueError("%s: %s, got %s" % (label, REFUSALS["global_scale_not_fp32"],
                                             tuple(tensor.shape)))
    return float(tensor.reshape(()).to(torch.float32))


class CompressedTensorsSource:
    """Read logical tensors of a ``compressed-tensors`` NVFP4 checkpoint.

    Parameters
    ----------
    model_dir:
        Directory holding either ``model.safetensors.index.json`` and its shards
        (a multi-shard export) or a single ``model.safetensors``.
    chunk_rows:
        Logical rows decoded per block in the dequantizing path.  The value only
        affects peak memory, never the returned words.
    """

    def __init__(self, model_dir: str | Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS):
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be at least one row")
        self.model_dir = Path(model_dir)
        self.chunk_rows = int(chunk_rows)

        self.weight_map: Mapping[str, str] = MappingProxyType(self._build_weight_map())
        self.keys: frozenset[str] = frozenset(self.weight_map)

        #: ``base`` -> the four stored key names it owns.
        self._groups: dict[str, dict[str, str]] = {}
        self._incomplete: list[str] = []
        aux: set[str] = set()
        for key in self.weight_map:
            if not key.endswith("." + FIELD_PACKED):
                continue
            base = key[: -(len(FIELD_PACKED) + 1)]
            fields = {FIELD_PACKED: key}
            complete = True
            for field in FIELDS[1:]:
                name = "%s.%s" % (base, field)
                if name not in self.weight_map:
                    complete = False
                    break
                fields[field] = name
            if not complete:
                self._incomplete.append(base)
                continue
            self._groups[base] = fields
            aux.update(fields[field] for field in FIELDS[1:])

        self._bases = tuple(sorted(self._groups))
        packed_names = {fields[FIELD_PACKED] for fields in self._groups.values()}
        self._names = tuple(
            sorted(
                list(self._bases)
                + [k for k in self.weight_map if k not in aux and k not in packed_names]
            )
        )

        self._handles: dict[str, Any] = {}
        self._metadata: dict[str, tuple[tuple[int, ...], str]] = {}

    # ------------------------------------------------------------------ index

    def _build_weight_map(self) -> dict[str, str]:
        """key -> shard name, from the index if present, else from the shards.

        A single-file export carries no index, and the work item that names this
        reader also names the index-aware path as part of it, so both spellings
        are read here rather than only the one this box happens to hold.
        """

        index_path = self.model_dir / INDEX_NAME
        if index_path.is_file():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("%s: weight_map must be a nonempty object" % index_path)
            return {str(k): str(v) for k, v in weight_map.items()}

        single = self.model_dir / SINGLE_SHARD_NAME
        candidates = [single] if single.is_file() else sorted(
            p for p in self.model_dir.glob(SHARD_GLOB) if p.is_file()
        )
        if not candidates:
            raise ValueError(
                "%s: neither %s nor any %s shard is present"
                % (self.model_dir, INDEX_NAME, SHARD_GLOB)
            )
        weight_map: dict[str, str] = {}
        for shard in candidates:
            header = _read_header_keys(shard)
            for key in header:
                if key in weight_map:
                    raise ValueError(
                        "%s: key %r appears in more than one shard" % (self.model_dir, key)
                    )
                weight_map[key] = shard.name
        return weight_map

    # ------------------------------------------------------------------- keys

    def names(self) -> tuple[str, ...]:
        """Every logical source key, sorted.

        The three NVFP4 side keys (``.weight_scale``, ``.weight_global_scale``,
        ``.input_global_scale``) are metadata rather than tensors and are
        excluded; the accessors below expose their content instead.
        """

        return self._names

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            "CompressedTensorsSource(%r, keys=%d, names=%d, quantized=%d, chunk_rows=%d, "
            "marker=%s)"
            % (str(self.model_dir), len(self.keys), len(self._names), len(self._bases),
               self.chunk_rows, COMPILED_MARKER)
        )

    def close(self) -> None:
        """Release every open shard handle; further reads raise."""

        for handle in self._handles.values():
            close = getattr(handle, "__exit__", None)
            if close is not None:
                close(None, None, None)
        self._handles.clear()

    def __enter__(self) -> "CompressedTensorsSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # --------------------------------------------------------------- plumbing

    def shard_of(self, key: str) -> str:
        try:
            return self.weight_map[key]
        except KeyError:
            raise KeyError("%r %s" % (key, REFUSALS["unknown_key"])) from None

    def _handle(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def metadata_of(self, key: str) -> tuple[tuple[int, ...], str]:
        """``(stored shape, stored dtype name)`` of one indexed key."""

        cached = self._metadata.get(key)
        if cached is not None:
            return cached
        descriptor = self._handle(self.shard_of(key)).get_slice(key)
        metadata = (
            tuple(int(dimension) for dimension in descriptor.get_shape()),
            str(descriptor.get_dtype()),
        )
        self._metadata[key] = metadata
        return metadata

    def _tensor(self, key: str) -> torch.Tensor:
        return self._handle(self.shard_of(key)).get_tensor(key)

    def _row_range(self, key: str, begin: int, count: int) -> torch.Tensor:
        descriptor = self._handle(self.shard_of(key)).get_slice(key)
        return descriptor[begin : begin + count]

    def _iter_blocks(self, rows: int) -> Iterator[tuple[int, int]]:
        step = max(1, min(self.chunk_rows, rows))
        for begin in range(0, rows, step):
            yield begin, min(step, rows - begin)

    # ------------------------------------------------------------- descriptor

    def _base_of(self, name: str) -> str:
        """Accept either the logical base name or ``base.weight_packed``."""

        if name.endswith("." + FIELD_PACKED):
            name = name[: -(len(FIELD_PACKED) + 1)]
        if name not in self._groups:
            if name in self.keys:
                raise KeyError("%r %s" % (name, REFUSALS["not_quantized"]))
            raise KeyError("%r %s" % (name, REFUSALS["unknown_key"]))
        return name

    @property
    def quantized_keys(self) -> tuple[str, ...]:
        """Every NVFP4 quantized base name in the checkpoint, sorted."""

        return self._bases

    @property
    def incomplete_groups(self) -> tuple[str, ...]:
        """``.weight_packed`` keys that do NOT carry all four fields, by name.

        Reported rather than skipped: a base with a code plane but no scale plane
        is a group whose payload would have to be invented, which is the exact
        outcome this reader exists to make auditable.
        """

        return tuple(sorted(self._incomplete))

    def fields_of(self, name: str) -> Mapping[str, str]:
        """The four stored key names of one quantized base."""

        return MappingProxyType(dict(self._groups[self._base_of(name)]))

    def is_nvfp4(self, name: str) -> bool:
        """Whether *name* is the base of a fully spelled NVFP4 group."""

        if name.endswith("." + FIELD_PACKED):
            name = name[: -(len(FIELD_PACKED) + 1)]
        return name in self._groups

    def shape_of(self, name: str) -> tuple[int, int]:
        """The logical ``(N, K)`` of one quantized base, checked against its words."""

        base = self._base_of(name)
        fields = self._groups[base]
        rows, half = self.metadata_of(fields[FIELD_PACKED])[0]
        scale_rows, scale_columns = self.metadata_of(fields[FIELD_SCALE])[0]
        column = half * 2
        if column % GROUP:
            raise ValueError("%s: %s (K=%d)" % (base, REFUSALS["group_size"], column))
        if (scale_rows, scale_columns) != (rows, column // GROUP):
            raise ValueError(
                "%s: %s -- packed %s implies (N, K/16) = %s, stored plane is %s"
                % (base, REFUSALS["scale_columns"], (rows, half),
                   (rows, column // GROUP), (scale_rows, scale_columns)))
        return rows, column

    # --------------------------------------------------------------- numeric

    def weight_global_scale(self, name: str) -> float:
        """The stored FP32 ``weight_global_scale`` -- a DIVISOR, as stored."""

        field = self._groups[self._base_of(name)][FIELD_WEIGHT_GLOBAL_SCALE]
        return _fp32_scalar(self._tensor(field), field)

    def input_global_scale(self, name: str) -> float:
        """The stored FP32 ``input_global_scale`` (a multiplier, unused here)."""

        field = self._groups[self._base_of(name)][FIELD_INPUT_GLOBAL_SCALE]
        return _fp32_scalar(self._tensor(field), field)

    def divisor(self, name: str) -> float:
        """``fp32(weight_global_scale)`` -- the engine's one trailing divisor.

        Deliberately NOT ``fp32(1 / weight_global_scale)``: this vocabulary stores
        a divisor while the modelopt vocabulary stores a multiplier, and the two
        differ by exactly one reciprocal (see the module docstring).
        """

        word = np.float32(self.weight_global_scale(name))
        if not np.isfinite(word) or word <= 0.0:
            raise ValueError("%s: %s" % (name, REFUSALS["global_scale_not_positive"]))
        return float(word)

    def divisor_word(self, name: str) -> bytes:
        """The four FP32 bytes ``layouts.encode_nvfp4`` takes as ``weight_divisor``."""

        return struct.pack("<f", np.float32(self.divisor(name)))

    def nvfp4_words(self, name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """The literal stored words of one quantized base.

        Returns ``(packed_codes, natural_scales, divisor)`` with

        * ``packed_codes`` -- ``uint8 (N, K/2)``; element ``2j`` is the low
          nibble of byte ``j`` and element ``2j+1`` its high nibble;
        * ``natural_scales`` -- ``uint8 (N, K/16)`` in natural (unswizzled) row
          order, holding the raw E4M3FN words rather than decoded floats;
        * ``divisor`` -- the FP32 global scale the engine divides by.

        Both tensors are detached copies, so they stay valid after :meth:`close`.
        """

        base = self._base_of(name)
        fields = self._groups[base]
        rows, column = self.shape_of(base)
        packed_meta = self.metadata_of(fields[FIELD_PACKED])
        scale_meta = self.metadata_of(fields[FIELD_SCALE])
        if packed_meta[1] not in _PACKED_DTYPES:
            raise ValueError(
                "%s: %s (%s)" % (fields[FIELD_PACKED], REFUSALS["packed_not_uint8"],
                                  packed_meta[1]))
        if scale_meta[1] not in ("F8_E4M3", "F8_E4M3FN"):
            raise ValueError(
                "%s: %s (%s)" % (fields[FIELD_SCALE], REFUSALS["scale_not_e4m3fn"],
                                  scale_meta[1]))
        if packed_meta[0] != (rows, column // 2):
            raise ValueError(
                "%s: %s -- stored %s, logical K %d implies %s"
                % (fields[FIELD_PACKED], REFUSALS["packed_columns"], packed_meta[0],
                   column, (rows, column // 2)))
        packed = self._tensor(fields[FIELD_PACKED]).reshape(rows, column // 2)
        scales = self._tensor(fields[FIELD_SCALE]).view(torch.uint8)
        scales = scales.reshape(rows, column // GROUP)
        _require_scale_words(scales, fields[FIELD_SCALE])
        divisor = self.divisor(base)
        return (
            packed.detach().to("cpu", copy=True).contiguous(),
            scales.detach().to("cpu", copy=True).contiguous(),
            divisor,
        )

    # --------------------------------------------------------------- logical

    def logical(
        self, name: str, device: str = "cpu", *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """The logical value of *name* as a tensor on *device*.

        NVFP4 bases dequantize to BF16 (override with ``dtype``); direct keys
        keep their stored dtype so the returned words stay bit-identical to the
        source.
        """

        result = (
            self._logical_nvfp4(name, dtype or torch.bfloat16)
            if self.is_nvfp4(name)
            else self._logical_direct(name, dtype)
        )
        return result.to(torch.device(device))

    def _logical_direct(self, name: str, dtype: torch.dtype | None) -> torch.Tensor:
        shape, stored_name = self.metadata_of(name)
        if dtype is not None:
            out_dtype: torch.dtype = dtype
        else:
            stored = _TORCH_DTYPES.get(stored_name)
            if stored is None:
                raise ValueError(
                    "%s: %s %r" % (name, REFUSALS["dtype_unsupported"], stored_name))
            out_dtype = stored
        if not shape:
            return self._tensor(name).to(out_dtype).reshape(())
        out = torch.empty(shape, dtype=out_dtype)
        for begin, count in self._iter_blocks(shape[0]):
            out[begin : begin + count] = self._row_range(name, begin, count)
        return out

    def _logical_nvfp4(self, name: str, out_dtype: torch.dtype) -> torch.Tensor:
        if not out_dtype.is_floating_point:
            raise TypeError("NVFP4 dequantization requires a floating-point dtype")
        base = self._base_of(name)
        fields = self._groups[base]
        rows, column = self.shape_of(base)
        groups = column // GROUP
        divisor = self.divisor(base)
        out = torch.empty((rows, column), dtype=out_dtype)
        for begin, count in self._iter_blocks(rows):
            packed = _flat_uint8(
                self._row_range(fields[FIELD_PACKED], begin, count), count, column // 2
            )
            scales = _flat_uint8(
                self._row_range(fields[FIELD_SCALE], begin, count).view(torch.uint8),
                count,
                groups,
            )
            _require_scale_words(torch.from_numpy(scales), fields[FIELD_SCALE])
            block = _dequantize_block(packed, scales, divisor)
            out[begin : begin + count] = torch.from_numpy(block).to(out_dtype)
        return out

    # ------------------------------------------------- artifact-layout bridge

    def encode_payload(
        self, name: str, shape: Sequence[int] | None = None, *, chunk_rows: int | None = None
    ) -> bytes:
        """Encode one object as the artifact's NVFP4 payload.

        This is the whole point of the reader: it turns this checkpoint's four
        fields into the bytes ``tools.artifact.layouts.encode_nvfp4`` owns.  The
        layout module is imported here rather than at module scope so the reader
        keeps working as a standalone source tool.
        """

        from tools.artifact.layouts import encode_nvfp4

        base = self._base_of(name)
        packed, scales, divisor = self.nvfp4_words(base)
        payload_shape = tuple(shape) if shape is not None else self.shape_of(base)
        return encode_nvfp4(packed, scales, self.divisor_word(base), payload_shape)

    # ---------------------------------------------------------------- census

    def self_check(self, plan: Any = None) -> dict[str, Any]:
        """The denominators this reader must reproduce, as its own reading.

        ``plan`` is optional: a mapping ``object name -> shape`` (or a sequence of
        ``(name, shape)`` pairs), keyed by the quantized base name.  Without it
        the source-side denominators are still complete; with it the plan-side
        ones are added.  Nothing here is asserted -- the caller gets the numbers
        and the *names* of whatever did not line up, and the verdict is its own.

        ``order_equal`` is deliberately not a tautology: it holds only when the
        plan declares exactly as many objects as the file carries, every declared
        object resolved to a distinct source group, and every group in the file
        was claimed by exactly one object.
        """

        index_entries = tuple(self.weight_map)
        matched: list[str] = []
        unresolved: list[tuple[str, str]] = []
        for key in index_entries:
            try:
                self.metadata_of(key)
            except Exception as exc:  # noqa: BLE001 -- reported by name, never hidden
                unresolved.append((key, "%s: %s" % (type(exc).__name__, exc)))
            else:
                matched.append(key)

        index_bases = tuple(
            k[: -(len(FIELD_PACKED) + 1)]
            for k in index_entries
            if k.endswith("." + FIELD_PACKED)
        )
        reports: dict[str, Any] = {
            "index_keys": len(index_entries),
            "matched": len(matched),
            "unmatched": tuple(unresolved),
            "set_equal": set(index_entries) == set(self.keys),
            "objects_in_file": len(self._bases),
            "packed_keys_in_index": len(index_bases),
            "incomplete_groups": self.incomplete_groups,
            "index_order_is_sorted": index_bases == tuple(sorted(index_bases)),
            "shards": tuple(sorted(set(self.weight_map.values()))),
        }

        geometry_bad: list[tuple[str, str]] = []
        for base in self._bases:
            try:
                self.shape_of(base)
                self.divisor(base)
            except Exception as exc:  # noqa: BLE001
                geometry_bad.append((base, "%s: %s" % (type(exc).__name__, exc)))
        reports["geometry_selfconsistent"] = (
            len(self._bases) - len(geometry_bad),
            len(self._bases),
        )
        reports["geometry_bad"] = tuple(geometry_bad)

        if plan is not None:
            items = (
                list(plan.items())
                if isinstance(plan, Mapping)
                else [(str(n), tuple(s)) for n, s in plan]
            )
            named = [str(name) for name, _ in items]
            mismatched: list[tuple[str, str]] = []
            for name, shape in items:
                if not self.is_nvfp4(name):
                    mismatched.append((name, "not an NVFP4 base of this checkpoint"))
                    continue
                logical = self.shape_of(name)
                if tuple(shape) != logical:
                    mismatched.append(
                        (name, "declared %s, source %s" % (tuple(shape), logical))
                    )
            reports["objects_declared"] = len(items)
            reports["plan_matched"] = len(items) - len(mismatched)
            reports["geometry_equal"] = (len(items) - len(mismatched), len(items))
            reports["geometry_mismatched"] = tuple(mismatched)
            reports["plan_names_distinct"] = len(set(named)) == len(named)
            reports["declared_not_in_file"] = tuple(sorted(set(named) - set(self._bases)))
            reports["in_file_not_declared"] = tuple(sorted(set(self._bases) - set(named)))
            reports["order_equal"] = (
                len(items) == len(self._bases)
                and len(set(named)) == len(named)
                and set(named) == set(self._bases)
                and not mismatched
            )
        return reports


def _read_header_keys(path: Path) -> tuple[str, ...]:
    """Tensor key names of one safetensors shard, from its header only."""

    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return tuple(header)


def _plan_from_inventory() -> dict[str, tuple[int, ...]] | None:
    """The register of NVFP4 objects this reader is asked to serve, if importable.

    Imported lazily and tolerated missing: the reader is a source tool and must
    not depend on one target being present.
    """

    try:
        from tools.convert.gemma4_31b import inventory, recipe
    except Exception:  # noqa: BLE001 -- a standalone source tool
        return None
    spec = {s.name: tuple(s.shape) for s in inventory.TEXT_TENSOR_SPECS}
    plan: dict[str, tuple[int, ...]] = {}
    for entry in recipe.OBJECT_RECIPES:
        if entry.op != "nvfp4-pass-through":
            continue
        first = entry.source_keys[0]
        if not first.endswith("." + FIELD_PACKED):
            continue
        plan[first[: -(len(FIELD_PACKED) + 1)]] = spec[entry.object_name]
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect a compressed-tensors NVFP4 source without an artifact or a GPU."""

    parser = argparse.ArgumentParser(
        prog="python3 -m tools.convert.dequant.compressed_tensors",
        description="Inspect a compressed-tensors NVFP4 HF source directory.",
    )
    parser.add_argument("--model", required=True, type=Path, help="source directory")
    parser.add_argument("--key", action="append", default=[],
                        help="quantized base to inspect; repeatable")
    parser.add_argument("--dequantize", action="store_true",
                       help="also decode each --key and report its binary32 range")
    parser.add_argument("--encode", action="append", default=[],
                        help="encode this base through layouts.encode_nvfp4 and round-trip it")
    parser.add_argument("--self-check", action="store_true",
                        help="print the denominators, including the plan when importable")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    arguments = parser.parse_args(argv)

    with CompressedTensorsSource(arguments.model, chunk_rows=arguments.chunk_rows) as source:
        print(source)
        print("  index keys        : %d" % len(source.keys))
        print("  logical names     : %d" % len(source.names()))
        print("  NVFP4 bases       : %d" % len(source.quantized_keys))

        if arguments.self_check:
            plan = _plan_from_inventory()
            print("  plan source       : %s" % ("gemma4_31b recipe" if plan else "absent"))
            for name, value in sorted(source.self_check(plan).items()):
                print("  %-24s: %s" % (name, value))

        for key in arguments.key:
            if not source.is_nvfp4(key):
                raise SystemExit("%r %s" % (key, REFUSALS["not_quantized"]))
            fields = source.fields_of(key)
            print("\n%s" % key)
            for field in FIELDS:
                shape, stored = source.metadata_of(fields[field])
                print("  %-20s: %-9s %s" % (field, stored, shape))
            print("  logical           : (N, K) = %s" % (source.shape_of(key),))
            print("  weight_global_scale: %r   (the DIVISOR, as stored)"
                  % source.weight_global_scale(key))
            print("  input_global_scale : %r" % source.input_global_scale(key))
            print("  divisor           : %r" % source.divisor(key))
            if arguments.dequantize:
                value = source.logical(key)
                print("  dequantized       : %s %s absmax=%.6g"
                      % (tuple(value.shape), value.dtype, float(value.abs().max())))

        for key in arguments.encode:
            from tools.artifact.layouts import decode_nvfp4_words

            payload = source.encode_payload(key)
            codes, scales, divisor = decode_nvfp4_words(payload, source.shape_of(key))
            packed, natural, _ = source.nvfp4_words(key)
            same_codes = bool(torch.equal(codes, packed))
            same_scales = bool(torch.equal(scales.view(torch.uint8), natural))
            print("\nencoded %s payload_bytes=%d round_trip_codes=%s round_trip_scales=%s "
                  "round_trip_divisor=%r" % (key, len(payload), same_codes, same_scales,
                                             float(divisor)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
