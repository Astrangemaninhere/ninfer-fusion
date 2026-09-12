"""Logical source reader for the ModelOpt NVFP4 HF checkpoint of Qwen3.8-27B.

The reader indexes a Hugging Face ``safetensors`` checkpoint directory and
exposes, per HF source key, either the stored words of a direct-format tensor
(BF16 / FP32) or the logical value of an NVFP4-quantized ``Linear`` weight
together with its *literal* stored words.

Measured storage contract of the source
---------------------------------------

* a quantized ``Linear`` owns exactly four keys:

  ==============================  ====================  ===========================
  key                             stored dtype          stored shape
  ==============================  ====================  ===========================
  ``<base>.weight``               ``U8``                ``(N, K/2)``
  ``<base>.weight_scale``         ``F8_E4M3``           ``(N, K/16)`` natural
  ``<base>.weight_scale_2``       ``F32``               scalar multiplier
  ``<base>.input_scale``          ``F32``               scalar multiplier
  ==============================  ====================  ===========================

* the scale plane is stored in **natural (unswizzled) row order**, so no
  permutation is needed to read it;
* element ``2j`` lives in the low nibble of packed byte ``j`` and element
  ``2j+1`` in its high nibble;
* ``W = E2M1(nibble) * E4M3FN(scale) * weight_scale_2``; the engine applies the
  same factor as a division by ``weight_divisor = fp32(1 / weight_scale_2)``,
  which :meth:`ModelOptSource.nvfp4_words` returns verbatim;
* every other key is a plain BF16 or FP32 tensor (norms, ``conv1d``, ``A_log``,
  ``dt_bias``, ``in_proj_a/b``, ``embed_tokens`` and all of ``model.visual.*``).

Nothing is loaded up front: construction only reads the safetensors index, and
every payload access is a row-range slice obtained from the shard handle, so a
single block of ``chunk_rows`` rows is the largest temporary the reader ever
allocates.  That matters because the source directory is 17.9 GB and its largest
quantized weight (``lm_head``) is a 2.4 GB logical matrix.

Canonical use::

    from tools.convert.dequant.modelopt import ModelOptSource

    with ModelOptSource("/models/q3nvfp4") as source:
        for key in source.names():
            weight = source.logical(key)
            literal = source.nvfp4_words(key)
            if literal is not None:
                packed, scales, divisor = literal

Dtype policy of :meth:`ModelOptSource.logical`
----------------------------------------------

NVFP4 weights dequantize to BF16 by default.  Direct keys keep their *stored*
dtype -- BF16 stays BF16 and FP32 stays FP32 -- so that the returned words are
bit-identical to the source; pass ``dtype=`` to force a uniform cast.

Standalone inspection (no artifact, no GPU)::

    python3 -m tools.convert.dequant.modelopt --model /models/q3nvfp4
    python3 -m tools.convert.dequant.modelopt --model /models/q3nvfp4 \\
        --key model.language_model.layers.3.mlp.gate_proj.weight --dequantize
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


__all__ = ["DEFAULT_CHUNK_ROWS", "ModelOptSource"]


DEFAULT_CHUNK_ROWS = 1024
"""Logical rows decoded per block; the reader's only peak-memory knob."""

INDEX_NAME = "model.safetensors.index.json"

_WEIGHT_SUFFIX = ".weight"
_AUX_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")
_GROUP = 16
"""Logical elements sharing one E4M3FN scale word along K."""

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

_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_code_bits() -> np.ndarray:
    """Binary32 word bits of every four-bit E2M1 code (bit 3 is the sign)."""

    values = [
        np.float32(
            np.copysign(
                _E2M1_MAGNITUDES[code & 0x7],
                -1.0 if code & 0x8 else 1.0,
            )
        )
        for code in range(16)
    ]
    return np.array(values, dtype=np.float32).view(np.uint32)


_E2M1_BITS = _e2m1_code_bits()


def _e2m1_pair_words() -> np.ndarray:
    """``(256,)`` table: one packed byte -> the two binary32 words it encodes.

    Entry ``b`` holds the binary32 bits of the low nibble in its low half and the
    bits of the high nibble in its high half.  On a little-endian host a single
    gather followed by a ``float32`` view therefore yields the logical element
    order ``[2j, 2j+1, 2j+2, ...]`` directly, with no strided scatter and no
    intermediate second gather.
    """

    codes = np.arange(256, dtype=np.uint32)
    low = _E2M1_BITS[codes & 0x0F].astype(np.uint64)
    high = _E2M1_BITS[codes >> 4].astype(np.uint64)
    return (low | (high << np.uint64(32))).astype(np.uint64)


_E2M1_PAIR_WORDS = _e2m1_pair_words()

_SUBNORMAL_BITS = np.array(
    [np.float32(fraction * 2.0**-9) for fraction in range(8)],
    dtype=np.float32,
).view(np.uint32)
"""Binary32 bits of ``fraction * 2**-9``, the E4M3FN subnormal range."""


def _e4m3fn_values(words: np.ndarray) -> np.ndarray:
    """Expand E4M3FN words to their exact binary32 values, by integer bits.

    The exponent and mantissa of a normal E4M3FN word map into binary32 without
    any rounding or lookup -- ``(8 + fraction) * 2 ** (exponent - 10)`` is exactly
    ``exponent_field = exponent + 120`` with ``fraction`` in the top three
    mantissa bits -- so the whole expansion is one vectorized integer pipeline.
    Only the eight subnormal codes need a table.

    ``0x7F`` / ``0xFF`` (the E4M3FN not-a-number encodings) are not represented
    faithfully; callers must screen them out first, which
    :func:`_require_scale_words` does for every NVFP4 scale plane.
    """

    word = np.ascontiguousarray(words, dtype=np.uint32)
    sign = (word & 0x80) << 24
    exponent = (word >> 3) & 0xF
    fraction = word & 0x7
    normal = ((exponent + 120) << 23) | (fraction << 20)
    bits = np.where(exponent == 0, _SUBNORMAL_BITS[fraction], normal)
    return (bits.astype(np.uint32) | sign).view(np.float32)


def _require_scale_words(words: torch.Tensor, key: str) -> None:
    """Reject E4M3FN scale words that are negative, non-finite, or ``0x7F``."""

    if bool((((words & 0x80) != 0) | (words == 0x7F)).any()):
        raise ValueError(
            f"{key}: NVFP4 scale words must be nonnegative finite E4M3FN codes"
        )


def _flat_uint8(tensor: torch.Tensor, rows: int, columns: int) -> np.ndarray:
    """Host ``(rows, columns)`` uint8 view of one raw-word block."""

    block = tensor.reshape(rows, columns).contiguous()
    return block.numpy()


def _dequantize_block(
    packed: np.ndarray, scales: np.ndarray, scale_2: float
) -> np.ndarray:
    """Vectorized ``E2M1(nibble) * E4M3FN(scale) * weight_scale_2`` in binary32."""

    rows, half = packed.shape
    column = half * 2
    groups = scales.shape[1]
    if column != groups * _GROUP:
        raise ValueError("packed codes and scale words disagree about the K axis")
    values = _E2M1_PAIR_WORDS[np.ascontiguousarray(packed)].view(np.float32)
    scale_values = _e4m3fn_values(scales)
    values = values.reshape(rows, groups, _GROUP) * scale_values[:, :, None]
    values *= np.float32(scale_2)
    return values.reshape(rows, column)


def _fp32_scalar(tensor: torch.Tensor, key: str) -> float:
    """Exact Python float of a stored FP32 scalar (a binary32 round-trip)."""

    if tensor.numel() != 1:
        raise ValueError(f"{key}: expected a scalar, got {tuple(tensor.shape)}")
    return float(tensor.reshape(()).to(torch.float32))


class ModelOptSource:
    """Read logical tensors of a ModelOpt-quantized HF checkpoint by source key.

    Parameters
    ----------
    model_dir:
        Directory holding ``model.safetensors.index.json`` and its shards.
    chunk_rows:
        Logical rows decoded per block.  Smaller values lower peak memory; the
        value only affects performance, never the returned words.
    """

    def __init__(self, model_dir: str | Path, *, chunk_rows: int = DEFAULT_CHUNK_ROWS):
        if chunk_rows < 1:
            raise ValueError("chunk_rows must be at least one row")
        self.model_dir = Path(model_dir)
        self.chunk_rows = int(chunk_rows)
        index_path = self.model_dir / INDEX_NAME
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path}: weight_map must be a nonempty object")
        self.weight_map: Mapping[str, str] = MappingProxyType(dict(weight_map))
        self.keys: frozenset[str] = frozenset(self.weight_map)
        self._names = tuple(
            sorted(key for key in self.keys if not key.endswith(_AUX_SUFFIXES))
        )
        self._handles: dict[str, Any] = {}
        self._metadata: dict[str, tuple[tuple[int, ...], str]] = {}

    # ------------------------------------------------------------------ keys

    def names(self) -> tuple[str, ...]:
        """Return every logical source key, sorted.

        The three NVFP4 side keys (``.weight_scale``, ``.weight_scale_2``,
        ``.input_scale``) are metadata rather than tensors and are excluded;
        :meth:`nvfp4_words`, :meth:`weight_scale_2` and :meth:`input_scale`
        expose their content instead.
        """

        return self._names

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"ModelOptSource({str(self.model_dir)!r}, keys={len(self.keys)}, "
            f"names={len(self._names)}, chunk_rows={self.chunk_rows})"
        )

    def close(self) -> None:
        """Release every open shard handle; further reads raise."""

        for handle in self._handles.values():
            close = getattr(handle, "__exit__", None)
            if close is not None:
                close(None, None, None)
        self._handles.clear()

    def __enter__(self) -> "ModelOptSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # -------------------------------------------------------------- plumbing

    def _handle(self, shard: str):
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def _shard_of(self, key: str) -> str:
        try:
            return self.weight_map[key]
        except KeyError:
            raise KeyError(f"{key!r} is not a key of this checkpoint") from None

    def _metadata_of(self, key: str) -> tuple[tuple[int, ...], str]:
        cached = self._metadata.get(key)
        if cached is not None:
            return cached
        descriptor = self._handle(self._shard_of(key)).get_slice(key)
        metadata = (
            tuple(int(dimension) for dimension in descriptor.get_shape()),
            str(descriptor.get_dtype()),
        )
        self._metadata[key] = metadata
        return metadata

    def _tensor(self, key: str) -> torch.Tensor:
        """Whole stored tensor of *key*, as a read-only view of the shard."""

        return self._handle(self._shard_of(key)).get_tensor(key)

    def _row_range(self, key: str, begin: int, count: int) -> torch.Tensor:
        """Stored rows ``[begin, begin + count)`` of *key*."""

        descriptor = self._handle(self._shard_of(key)).get_slice(key)
        return descriptor[begin : begin + count]

    def _iter_blocks(self, rows: int) -> Sequence[tuple[int, int]]:
        step = max(1, min(self.chunk_rows, rows))
        return tuple(
            (begin, min(step, rows - begin)) for begin in range(0, rows, step)
        )

    @property
    def quantized_keys(self) -> tuple[str, ...]:
        """Return every NVFP4 ``.weight`` key in the checkpoint, sorted."""

        return tuple(key for key in self._names if self.is_nvfp4(key))

    # ------------------------------------------------------------ descriptor

    def is_nvfp4(self, key: str) -> bool:
        """Return whether *key* is the ``.weight`` of a quantized ``Linear``."""

        if key not in self.keys or not key.endswith(_WEIGHT_SUFFIX):
            return False
        base = key[: -len(_WEIGHT_SUFFIX)]
        if base + ".weight_scale" not in self.keys:
            return False
        if base + ".weight_scale_2" not in self.keys:
            return False
        return self._metadata_of(key)[1] == "U8"

    def shape_of(self, key: str) -> tuple[int, ...]:
        """Return the logical shape: ``(N, K)`` for NVFP4, the stored shape else."""

        shape, _ = self._metadata_of(key)
        if not self.is_nvfp4(key):
            return shape
        if len(shape) != 2:
            raise ValueError(f"{key}: an NVFP4 weight must be rank 2, got {shape}")
        groups = self._metadata_of(self._base_of(key) + ".weight_scale")[0][-1]
        return (shape[0], groups * _GROUP)

    def _base_of(self, key: str) -> str:
        if not self.is_nvfp4(key):
            raise KeyError(f"{key!r} is not an NVFP4-quantized weight")
        return key[: -len(_WEIGHT_SUFFIX)]

    def weight_scale_2(self, key: str) -> float:
        """Return the FP32 ``weight_scale_2`` multiplier of a quantized weight."""

        name = self._base_of(key) + ".weight_scale_2"
        return _fp32_scalar(self._tensor(name), name)

    def input_scale(self, key: str) -> float:
        """Return the FP32 ``input_scale`` multiplier of a quantized weight."""

        name = self._base_of(key) + ".input_scale"
        if name not in self.keys:
            raise KeyError(f"{name!r} is not a key of this checkpoint")
        return _fp32_scalar(self._tensor(name), name)

    # ---------------------------------------------------------------- literal

    def nvfp4_words(
        self, key: str
    ) -> tuple[torch.Tensor, torch.Tensor, float] | None:
        """Return the literal stored words of an NVFP4 weight, else ``None``.

        The result is ``(packed_codes, natural_scales, divisor)`` with

        * ``packed_codes`` -- ``uint8 (N, K/2)``; element ``2j`` is the low
          nibble of byte ``j`` and element ``2j+1`` its high nibble;
        * ``natural_scales`` -- ``uint8 (N, K/16)``, natural (unswizzled) row
          order, holding the raw E4M3FN words rather than decoded floats;
        * ``divisor`` -- the FP32 ``1 / weight_scale_2`` the engine divides by.

        Both tensors are detached copies, so they stay valid after
        :meth:`close`.  Non-quantized keys return ``None``.
        """

        if not self.is_nvfp4(key):
            return None
        rows, column = self.shape_of(key)
        base = self._base_of(key)
        groups = column // _GROUP
        packed = self._tensor(key).reshape(rows, column // 2)
        scales = self._tensor(base + ".weight_scale").view(torch.uint8)
        scales = scales.reshape(rows, groups)
        _require_scale_words(scales, key)
        scale_2 = self.weight_scale_2(key)
        divisor = float(np.float32(1.0) / np.float32(scale_2))
        return (
            packed.detach().to("cpu", copy=True).contiguous(),
            scales.detach().to("cpu", copy=True).contiguous(),
            divisor,
        )

    # ---------------------------------------------------------------- logical

    def logical(
        self,
        key: str,
        device: str = "cpu",
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the logical value of *key* as a tensor on *device*.

        NVFP4 weights are dequantized to BF16 (override with ``dtype``).  Direct
        keys keep their stored dtype so that the returned words stay
        bit-identical to the source; pass ``dtype=torch.bfloat16`` to force a
        uniform cast.
        """

        if self.is_nvfp4(key):
            result = self._logical_nvfp4(key, dtype or torch.bfloat16)
        else:
            result = self._logical_direct(key, dtype)
        return result.to(torch.device(device))

    def _logical_direct(self, key: str, dtype: torch.dtype | None) -> torch.Tensor:
        shape, stored_name = self._metadata_of(key)
        if dtype is not None:
            out_dtype: torch.dtype = dtype
        else:
            stored = _TORCH_DTYPES.get(stored_name)
            if stored is None:
                raise ValueError(f"{key}: unsupported stored dtype {stored_name!r}")
            out_dtype = stored
        if not shape:
            return self._tensor(key).to(out_dtype).reshape(())
        out = torch.empty(shape, dtype=out_dtype)
        for begin, count in self._iter_blocks(shape[0]):
            out[begin : begin + count] = self._row_range(key, begin, count)
        return out

    def _logical_nvfp4(self, key: str, out_dtype: torch.dtype) -> torch.Tensor:
        if not out_dtype.is_floating_point:
            raise TypeError("NVFP4 dequantization requires a floating-point dtype")
        rows, column = self.shape_of(key)
        base = self._base_of(key)
        groups = column // _GROUP
        scale_2 = self.weight_scale_2(key)
        out = torch.empty((rows, column), dtype=out_dtype)
        for begin, count in self._iter_blocks(rows):
            packed = _flat_uint8(
                self._row_range(key, begin, count), count, column // 2
            )
            scales = _flat_uint8(
                self._row_range(base + ".weight_scale", begin, count).view(torch.uint8),
                count,
                groups,
            )
            _require_scale_words(torch.from_numpy(scales), key)
            block = _dequantize_block(packed, scales, scale_2)
            out[begin : begin + count] = torch.from_numpy(block).to(out_dtype)
        return out


def main(argv: Sequence[str] | None = None) -> None:
    """Inspect a ModelOpt source directory without an artifact or a GPU."""

    parser = argparse.ArgumentParser(
        prog="python3 -m tools.convert.dequant.modelopt",
        description="Inspect a ModelOpt NVFP4 HF source directory.",
    )
    parser.add_argument("--model", required=True, type=Path, help="source directory")
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="source key to inspect; repeatable",
    )
    parser.add_argument(
        "--dequantize",
        action="store_true",
        help="also decode each --key and report its binary32 range",
    )
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    arguments = parser.parse_args(argv)

    with ModelOptSource(arguments.model, chunk_rows=arguments.chunk_rows) as source:
        print(source)
        quantized = source.quantized_keys
        print(f"  index keys        : {len(source.keys)}")
        print(f"  logical names     : {len(source.names())}")
        print(f"  NVFP4 weights     : {len(quantized)}")
        print(f"  first name        : {source.names()[0]}")
        print(f"  last name         : {source.names()[-1]}")
        for key in arguments.key:
            if key not in source:
                raise SystemExit(f"{key!r} is not a key of this checkpoint")
            shape, stored_name = source._metadata_of(key)
            print(f"\n{key}")
            print(f"  stored            : {stored_name} {shape}")
            literal = source.nvfp4_words(key)
            if literal is None:
                print("  logical           : {}".format(source.shape_of(key)))
            else:
                packed, scales, divisor = literal
                print(f"  logical           : (N, K) = {source.shape_of(key)}")
                print(f"  packed codes      : {tuple(packed.shape)} uint8")
                print(f"  natural scales    : {tuple(scales.shape)} uint8")
                print(f"  weight_scale_2    : {source.weight_scale_2(key)!r}")
                print(f"  divisor           : {divisor!r}")
                print(f"  input_scale       : {source.input_scale(key)!r}")
            if arguments.dequantize:
                value = source.logical(key)
                print(
                    f"  dequantized       : {tuple(value.shape)} {value.dtype} "
                    f"absmax={float(value.abs().max()):.6g}"
                )


if __name__ == "__main__":
    main()
