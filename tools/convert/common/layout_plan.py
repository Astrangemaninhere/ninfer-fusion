#!/usr/bin/env python3
"""Structural layout selection for imports: a source tensor group -> artifact layout.

WHY THIS EXISTS
---------------
The registered converters are closed, byte-pinned contracts: each hardcodes the
object plan of ONE checkpoint, which is correct but cannot answer "what should
this new source become?".  The front door (``import_model.py``) already reports
*that* a converter is missing; this module is the missing half - it decides the
layout from the source's own structure.

WHAT IT KEYS ON (and what it must NOT key on)
---------------------------------------------
Only two things, both taken from the source itself:

* the **dtype/shape relationship inside one quantised group** (the codes and the
  companion scales of one Linear), and
* the source's **declared quantisation algorithm** (ModelOpt writes
  ``hf_quant_config.json`` with a per-layer ``quant_algo``), used as a
  cross-check, never as the sole reason.

It deliberately contains **no model name, no layer name and no key allowlist**
beyond the suffix convention the quantiser itself defines (``.weight``,
``.weight_scale``, ``.weight_scale_2``): a new family is a data question, not a
code change.  Anything not recognised is refused with the missing mechanism
named, never approximated.

THE CONSTRAINTS COME FROM THE LAYOUTS, NOT FROM HERE
----------------------------------------------------
``tools/artifact/layouts.py`` is the authority:

* ``blockscale-k16-m128x4-v1`` (NVFP4) raises unless ``n % 128 == 0`` and
  ``k % 64 == 0``; it stores ``n*k/2`` code bytes, ``n*k/16`` scale bytes and a
  trailing fp32 ``weight_divisor``.
  ``encode_nvfp4`` moves the source words **without numerical conversion**, so
  an NVFP4 source is a pure repack.
* ``row-scale-v1`` (``FP8_E4M3FN_ROW_BF16S``) stores ``n*k`` code bytes and
  ``n*2`` scale bytes, i.e. **one BF16 scale per row**, with no divisibility
  requirement.  A per-tensor (scalar) source scale is therefore the degenerate
  case of a per-row scale and is materialised by broadcasting; the fp32 -> bf16
  narrowing is reported as a deviation instead of being hidden.
* ``contiguous-le-v1`` carries BF16/FP32/I32 verbatim.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

# Layout names and the encoders that fill them (from tools/artifact/layouts.py).
LAYOUT_CONTIGUOUS = "contiguous-le-v1"
LAYOUT_BLOCKSCALE = "blockscale-k16-m128x4-v1"
LAYOUT_ROW_SPLIT = "row-split-k128-v1"
LAYOUT_ROW_SCALE = "row-scale-v1"

ENCODER_DIRECT = "layouts.encode_direct"
ENCODER_NVFP4 = "layouts.encode_nvfp4"
ENCODER_FP8_ROW = "layouts.encode_fp8_row_scaled"

#: Layout constraints, quoted from the geometry functions so a drift there
#: becomes a failing test here rather than a silently wrong object plan.
BLOCKSCALE_N_MULTIPLE = 128
BLOCKSCALE_K_MULTIPLE = 64
NVFP4_GROUP = 16

#: dtypes that go through as raw bytes.
DIRECT_DTYPES = frozenset({"BF16", "FP32", "I32", "F32", "F16"})


class TensorMeta(NamedTuple):
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Choice:
    """A recognised group, with the layout that will hold it."""

    layout: str
    encoder: str
    reason: str
    deviations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Refusal:
    """An unrecognised/inexpressible group: name the missing mechanism."""

    code: str
    reason: str
    missing_mechanism: str

    @property
    def ok(self) -> bool:
        return False


def _canonical_dtype(dtype: str) -> str:
    d = dtype.upper().replace("-", "_")
    return {"FLOAT32": "FP32", "F32": "F32", "BFLOAT16": "BF16"}.get(d, d)


def _scale_rows(meta: TensorMeta) -> int | str:
    """How many row scales a source scale tensor carries: an int, or a label."""
    if len(meta.shape) == 0:
        return "scalar"
    if len(meta.shape) == 1:
        return int(meta.shape[0])
    if len(meta.shape) == 2:
        return int(meta.shape[0])
    return "unsupported"


def select(members: Mapping[str, TensorMeta], declared_algo: str | None = None) -> Choice | Refusal:
    """Decide the artifact layout for one quantised (or plain) group.

    ``members`` maps the *last path segment* (e.g. ``weight``, ``weight_scale``)
    of one module's tensors to its dtype/shape.  Grouping by module prefix is the
    caller's job because only the caller knows the naming convention of the
    source; everything below is decided from structure.
    """
    codes = members.get("weight")
    scale = members.get("weight_scale") or members.get("scale")
    scale2 = members.get("weight_scale_2")
    algo = (declared_algo or "").upper()

    # --- packed sub-8-bit integer codes (the ``pack-quantized`` spelling) ----
    # The code plane is named ``weight_packed`` and its own dtype is a 32-bit
    # integer word, which is in DIRECT_DTYPES; a group like that must not be
    # carried verbatim just because its words happen to be a directly
    # representable dtype.  No layout in tools/artifact/layouts.py holds packed
    # integer codes, so the missing mechanism is named here.  The code plane is
    # read by tools/convert/dequant/int3.py, which takes the packing unit from
    # the checkpoint's own declaration rather than from a key or a directory
    # name.
    packed = members.get("weight_packed")
    if packed is not None and (
        members.get("weight_scale") is not None
        or members.get("weight_shape") is not None
    ):
        packed_dtype = _canonical_dtype(packed.dtype)
        companions = sorted(
            name
            for name in ("weight_scale", "weight_shape", "weight_zero_point",
                         "weight_g_idx", "input_global_scale")
            if members.get(name) is not None
        )
        detail = (
            f"packed code plane {packed_dtype}{tuple(packed.shape)} with "
            f"companions {companions}"
        )
        shape_meta = members.get("weight_shape")
        if shape_meta is not None and len(shape_meta.shape) == 1:
            detail += f"; declared logical shape {tuple(shape_meta.shape)}"
        return Refusal(
            "F-PACKED-INT",
            detail,
            "a layout for packed sub-8-bit integer codes, plus the packing "
            "unit: layouts.py holds only whole-word layouts "
            "(contiguous-le-v1 / blockscale-k16-m128x4-v1 / row-scale-v1) and "
            "no packed-integer geometry, and the unit itself must come from "
            "the checkpoint's declaration (tools/convert/dequant/int3.py "
            "reads it; the front door would have to pass it here)",
        )


    # --- dispatch on the CODE tensor's own dtype, not on the presence of a
    #     scale companion: a plain BF16/FP32 tensor has nothing to decode, and
    #     deciding that first is what keeps ordinary modules (embeddings, norms,
    #     lm_head) out of the quantiser branches.
    code_dtype = _canonical_dtype(codes.dtype) if codes is not None else None

    if codes is None or code_dtype in DIRECT_DTYPES:
        dtypes = {_canonical_dtype(m.dtype) for m in members.values()}
        unknown = dtypes - DIRECT_DTYPES
        if unknown:
            return Refusal(
                "F-UNSUPPORTED-DTYPE",
                f"group carries {sorted(unknown)} and no quantisation companion",
                "a source format for these dtypes (add to tools/artifact/layouts.py "
                "_DIRECT_DTYPES or supply a codec)",
            )
        return Choice(
            LAYOUT_CONTIGUOUS,
            ENCODER_DIRECT,
            f"verbatim {sorted(dtypes)} ({'no code tensor' if codes is None else 'code dtype is direct'})",
        )

    # --- NVFP4: codes U8 [N, K/2] with a F8 scale [N, K/16] ----------------
    # Structural invariant: scale columns * (group/2) == code columns, because
    # K/2 code bytes vs K/16 scale bytes is a fixed ratio of 8 for group 16.
    if code_dtype == "U8" and scale is not None:
        sd = _canonical_dtype(scale.dtype)
        if len(codes.shape) == 2 and len(scale.shape) == 2:
            n, code_cols = codes.shape
            s_n, s_cols = scale.shape
            ratio_ok = code_cols == s_cols * (NVFP4_GROUP // 2)
            if sd == "F8_E4M3" and ratio_ok and n == s_n:
                if n % BLOCKSCALE_N_MULTIPLE or (code_cols * 2) % BLOCKSCALE_K_MULTIPLE:
                    return Refusal(
                        "F-LAYOUT-CONSTRAINT",
                        f"NVFP4 group {n}x{code_cols * 2} violates "
                        f"blockscale-k16-m128x4-v1 (n%{BLOCKSCALE_N_MULTIPLE}==0, "
                        f"k%{BLOCKSCALE_K_MULTIPLE}==0)",
                        "a padded or sub-tiled variant of blockscale-k16-m128x4-v1 "
                        "(tools/artifact/layouts.py block_scale_geometry)",
                    )
                devs: list[str] = []
                if scale2 is not None:
                    devs.append(
                        "per-tensor weight_scale_2 multiplier is stored as the layout's "
                        "trailing fp32 weight_divisor (the engine divides by it)"
                    )
                if algo and algo not in ("NVFP4", "MIXED_PRECISION"):
                    devs.append(f"declared quant_algo={declared_algo} disagrees with the NVFP4 structure")
                return Choice(
                    LAYOUT_BLOCKSCALE,
                    ENCODER_NVFP4,
                    f"U8[{n},{code_cols}] codes with F8_E4M3[{s_n},{s_cols}] block scales "
                    f"(codes/scales columns ratio {NVFP4_GROUP // 2})",
                    tuple(devs),
                )

    # --- FP8 row scale: codes F8 [N, K] + a scale of [] / [N] / [N, 1] -----
    if code_dtype == "F8_E4M3" and scale is not None:
        sd = _canonical_dtype(scale.dtype)
        if sd in ("F32", "FP32", "BF16", "F16"):
            rows = _scale_rows(scale)
            if rows == "scalar":
                shape_dev = (
                    "per-tensor fp32 scale is broadcast to the layout's per-row BF16 "
                    "scales; the narrowing fp32->bf16 is the only numerical change and "
                    "the codes themselves are moved verbatim"
                )
            elif isinstance(rows, int) and len(codes.shape) == 2 and rows == codes.shape[0]:
                shape_dev = "per-row scale taken directly"
            else:
                return Refusal(
                    "F-SCALE-SHAPE",
                    f"FP8 scale shape {tuple(scale.shape)} does not describe a per-row scale for codes "
                    f"{tuple(codes.shape)}",
                    "a scale layout for this shape (row-scale-v1 stores exactly one "
                    "BF16 multiplier per row)",
                )
            devs = [shape_dev]
            if algo and algo not in ("FP8", "MIXED_PRECISION"):
                devs.append(f"declared quant_algo={declared_algo} disagrees with the FP8 structure")
            n, k = (codes.shape if len(codes.shape) == 2 else (0, 0))
            return Choice(
                LAYOUT_ROW_SCALE,
                ENCODER_FP8_ROW,
                f"F8_E4M3 codes [{n},{k}] with a {rows} {sd} scale",
                tuple(devs),
            )

    if scale is None:
        return Refusal(
            "F-UNRECOGNISED-GROUP",
            f"codes {code_dtype}{tuple(codes.shape)} with no quantisation companion, and "
            f"{code_dtype} is not a directly representable dtype",
            f"a codec for {code_dtype} codes (tools/artifact/layouts.py holds no layout for them, "
            f"and _DIRECT_DTYPES does not carry the dtype either)",
        )
    return Refusal(
        "F-UNRECOGNISED-GROUP",
        f"codes {code_dtype}{tuple(codes.shape)} with a {_canonical_dtype(scale.dtype)}"
        f"{tuple(scale.shape)} scale, which no registered layout describes",
        "a codec for this (codes dtype, scale dtype, scale shape) combination",
    )


def group_by_module(names: Iterable[str]) -> dict[str, dict[str, TensorMeta]]:
    """Split a flat census into per-module groups keyed by the last segment.

    The split point is the LAST dot: everything before it is the module, the
    remainder is the field.  That is true for every HF checkpoint shape seen so
    far and needs no per-family knowledge.
    """
    groups: dict[str, dict[str, TensorMeta]] = {}
    for name, meta in names.items():
        module, _, fieldname = name.rpartition(".")
        if not module:
            module, fieldname = "", name
        groups.setdefault(module, {})[fieldname] = meta
    return groups


def load_census(path: Path) -> dict[str, TensorMeta]:
    raw = json.loads(path.read_text())
    return {k: TensorMeta(v["dtype"], tuple(v["shape"])) for k, v in raw.items()}


def load_declared_algos(source: Path) -> dict[str, str]:
    """Per-layer ``quant_algo`` as the source itself declares it."""
    hf = source / "hf_quant_config.json"
    if not hf.is_file():
        return {}
    doc = json.loads(hf.read_text())
    inner = doc.get("quantization") or {}
    layers = inner.get("quantized_layers") or {}
    return {k: str(v.get("quant_algo", "")).upper() for k, v in layers.items() if isinstance(v, Mapping)}


def plan_census(census: Mapping[str, TensorMeta],
                declared: Mapping[str, str]) -> tuple[dict[str, int], list[tuple[str, Refusal]]]:
    """Plan every module in a census; return (layout histogram, refusals)."""
    hist: dict[str, int] = {}
    refusals: list[tuple[str, Refusal]] = []
    for module, members in sorted(group_by_module(census).items()):
        algo = None
        for key, value in declared.items():
            if module == key or module.endswith(key) or key.endswith(module):
                algo = value
                break
        decision = select(members, algo)
        if decision.ok:
            hist[decision.layout] = hist.get(decision.layout, 0) + 1
        else:
            refusals.append((module, decision))
    return hist, refusals


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: layout_plan.py <_census_tensors.json> <source_dir>")
        return 2
    census = load_census(Path(args[0]))
    declared = load_declared_algos(Path(args[1]))
    hist, refusals = plan_census(census, declared)
    print(f"census: {len(census)} tensors, {len(declared)} declared quant_algo entries")
    print("layout histogram:")
    for layout, count in sorted(hist.items(), key=lambda kv: -kv[1]):
        print(f"  {layout:<28} {count} groups")
    if refusals:
        print(f"refusals: {len(refusals)}")
        shown: dict[str, int] = {}
        for module, r in refusals:
            shown[r.code] = shown.get(r.code, 0) + 1
            if shown[r.code] <= 3:
                print(f"  [{r.code}] {module}")
                print(f"      reason  : {r.reason}")
                print(f"      missing : {r.missing_mechanism}")
        print("  refusal histogram:", shown)
    else:
        print("refusals: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
