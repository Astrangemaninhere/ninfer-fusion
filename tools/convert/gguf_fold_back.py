#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_fold_back.py — ROUTE B, and the route SELECTOR for a rotated GGUF.

THE TWO ROUTES, AND WHY THERE ARE TWO
-------------------------------------
A Prism/Bonsai rotated checkpoint stores 402 tensors in the basis of ``A = H . D`` and declares
it under ``prism.hadamard.*``.  The map back into the primal basis must be applied EXACTLY ONCE,
and there are two honest places to apply it:

  Route A  ``pass_through``  The converter touches no tensor and DECLARES the checkpoint
                             ``prism.fold.basis = folded``.  The runtime applies ``A`` to the
                             activation of every declared weight and permutes the activation of
                             ``blk.*.ssm_out.weight``.  Needs ``src/ops/linear/prism_fold/``.
  Route B  ``fold_back``     The converter applies ``A^-1`` to all 402 declared tensors and
                             unpermutes the 48 GDN outputs to tiled order, so the emitted
                             checkpoint is ``primal`` and needs NO runtime transform.

Both are correct.  Both at once is a SILENT WRONG: ``A`` is an orthonormal involution, so
``A^-1 A = I`` and the second application cancels the first, leaving exactly the answer you get
by applying neither.  Measured on the real file, "twice" and "zero times" agree to fp32 roundoff
while differing from the correct answer by 1.35 x |y|max.  The two routes are therefore
**selectable and mutually exclusive by construction**:

  * the route is a SINGLE-VALUED enum on one command line, so no invocation can ask for both;
  * admission is decided against a **LEDGER OF WHAT THE ARITHMETIC ACTUALLY DID**, never
    against the flag's value.  ``fold_back`` is admitted only when every declared tensor was
    unfolded AND the round trip verified; ``pass_through`` only when the ledger is EMPTY.  The
    two requirements are disjoint, so an execution cannot satisfy both, so a checkpoint cannot
    be emitted by a run that half-folded;
  * the route is DERIVED FROM THE SOURCE when the operator names none: a file that declares
    its weights are folded gets ``fold_back``, a file that declares no fold gets
    ``pass_through``.  ``refuse`` -- emit nothing -- is a value an operator types, never an
    answer the tool arrives at by itself;
  * turning a route off is ``refuse`` or ``pass_through``-on-a-primal-source: both REFUSE.  No
    flag value makes the guard more permissive than its arithmetic.

The decision itself lives in :mod:`tools.convert.gguf_fold_route` and is mirrored, cell for
cell, in ``src/ops/linear/prism_fold/prism_fold.h``.

WHERE THE BASIS IS DECLARED
---------------------------
In the safetensors header's ``__metadata__`` map, and in ``config.json`` -- under the SAME key
spellings (``prism.fold.*``, and ``gguf_fold_route.BASIS_KEY`` for the basis itself), from the
same three inputs, so a consumer that reads one and a consumer that reads the other cannot
come to different answers about which basis the bytes are in.  ``__metadata__`` first: it is
the format's own metadata slot, not
an invented one, and this tree's readers already tolerate it — `lora_merge.py:132`,
`qwen3_8_flash_next/source.py:96`, `qwen3_8_27b/convert_modelopt.py:290` and `mtp.py:97` all
skip that key by name (G1/MEASURED, grepped).  A ``folded`` checkpoint that cannot declare
itself is therefore INADMISSIBLE by construction rather than by convention: without the
declaration the runtime cannot tell a folded file from an unrotated one of the same family, and
for this family those two differ in one tensor's inner head order under the same name.

WHAT THIS FILE DOES NOT DO
--------------------------
It does not implement Route A's runtime half, and it does not pretend the emitted ``folded``
checkpoint can be run by this engine today: ``kPrismFoldRuntimeSupportsTransform`` is ``false``
in every engine TU, so the engine REFUSES a ``folded`` checkpoint by name.  ``pass_through``
here is the converter half of Route A, delivered and exercised; the runtime half is delivered as
source and proven only to COMPILE, for the reason recorded in ``prism_fold_launch.h``.

WHY THE PLAN AND WRITE LOOPS ARE RE-SPELLED FROM gguf_extract.py
---------------------------------------------------------------
``gguf_extract.plan_tensors`` refuses a rotated file outright — that refusal is the fold guard,
and it is deliberately route-blind.  ``gguf_extract.write_safetensors`` has no hook between
``to_fp32`` and ``fp32_to_bf16_bytes``, which is the only place the fold-back can go.  Editing
either in place would collide with a LIVE line: `dl/type41land` was measured (2026-09-20
01:07) executing exactly that file, `python3 .../gguf_extract.py --src ...Bonsai-27B-Q1_0.gguf`,
pid 349209, 1288 s into a 5400 s timeout.  Everything else is imported: the name rules
(``gguf_names``), the codec table (``gguf_kquant``), the contract arithmetic
(``gguf_hadamard``), the decision (``gguf_fold_route``), and the output transaction and config
builder (``gguf_extract.OutputTransaction`` / ``.config_from_metadata``).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.convert import gguf_extract as X      # noqa: E402  OutputTransaction, config
from tools.convert import gguf_fold_route as FR   # noqa: E402  the decision
from tools.convert import gguf_hadamard as H      # noqa: E402  the arithmetic
from tools.convert import gguf_kquant as K        # noqa: E402  the codec table
from tools.convert import gguf_names as N         # noqa: E402  the name rules
from tools.convert import unit_offset_norms    # noqa: E402  the +1 the epilogue adds


class EmissionRefused(RuntimeError):
    """Raised with the missing part named; the caller leaves ZERO files."""


#: Element widths of the GGUF metadata types, for the value reader below.  Types 8 (string) and
#: 9 (array) are variable-length and are handled explicitly rather than by a table entry.
_GGUF_TYPE_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_GGUF_INT_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
                 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def read_metadata_values(path, wanted_keys):
    """The REAL values of ``wanted_keys``, read from the file's own metadata.

    ``gguf_kquant.read_tensor_table`` deliberately summarises arrays above its threshold — an
    element type, a count and a few leading values, in an object that is not iterable by design
    (``GGUFArray``) — and the contract's arrays are far above it: ``prism.hadamard.sign_values``
    is 28,672 entries.  ``gguf_extract.py``'s fold guard records this and avoids re-parsing the
    contract for exactly this reason; Route B cannot avoid it, because the unfold needs the sign
    VALUES and not their count.  So this reads them, and only them: every key that is not wanted
    is stepped over by its own encoded length, which is why ``tokenizer.ggml.tokens`` (248,320
    strings, ~200 MB) is skipped rather than materialised.

    Returns ``{key: value}`` for the keys actually present.  Values are Python ints/floats/strs,
    or lists for arrays.  A file this cannot walk raises :class:`EmissionRefused` rather than
    yielding a partial dict, because a partially-read contract is the "reading half a file"
    failure ``gguf_hadamard.UnreadableMetadata`` exists to name.
    """
    wanted = set(wanted_keys)
    out = {}
    try:
        with open(path, "rb") as fh:
            magic, _ver, _n_tensors, n_kv = struct.unpack("<4sIQQ", fh.read(24))
            if magic != b"GGUF":
                raise EmissionRefused("%s is not a GGUF file (magic %r)" % (path, magic))

            def rd_str():
                (n,) = struct.unpack("<Q", fh.read(8))
                return fh.read(n).decode("utf-8", "replace")

            def skip_str():
                (n,) = struct.unpack("<Q", fh.read(8))
                fh.seek(n, os.SEEK_CUR)

            def skip_array():
                at, cnt = struct.unpack("<IQ", fh.read(12))
                if at == 8:                       # array of strings: per-element lengths
                    for _ in range(cnt):
                        skip_str()
                elif at in _GGUF_TYPE_BYTES:
                    fh.seek(_GGUF_TYPE_BYTES[at] * cnt, os.SEEK_CUR)
                else:
                    raise EmissionRefused("metadata array element type %d not handled" % at)

            def rd_value(t, take):
                if t == 8:
                    return rd_str() if take else (skip_str(), None)[1]
                if t in _GGUF_INT_FMT and t != 6:
                    return struct.unpack(_GGUF_INT_FMT[t], fh.read(_GGUF_TYPE_BYTES[t]))[0]
                if t == 6:
                    return struct.unpack("<f", fh.read(4))[0]
                raise EmissionRefused("metadata type %d not handled" % t)

            for _ in range(n_kv):
                key = rd_str()
                (t,) = struct.unpack("<I", fh.read(4))
                take = key in wanted
                if t == 9:
                    at, cnt = struct.unpack("<IQ", fh.read(12))
                    if not take:
                        # step over it: strings individually, scalars by width
                        if at == 8:
                            for _ in range(cnt):
                                skip_str()
                        elif at in _GGUF_TYPE_BYTES:
                            fh.seek(_GGUF_TYPE_BYTES[at] * cnt, os.SEEK_CUR)
                        else:
                            raise EmissionRefused(
                                "metadata array element type %d not handled" % at)
                    else:
                        if cnt > (1 << 24):
                            raise EmissionRefused(
                                "metadata key %r declares %d elements; refused as implausible "
                                "rather than materialised" % (key, cnt))
                        if at == 8:
                            out[key] = [rd_str() for _ in range(cnt)]
                        elif at in _GGUF_INT_FMT:
                            raw = fh.read(_GGUF_TYPE_BYTES[at] * cnt)
                            out[key] = list(struct.unpack("<%d%s" % (cnt, _GGUF_INT_FMT[at][1:]),
                                                          raw))
                        else:
                            raise EmissionRefused(
                                "metadata array element type %d not handled" % at)
                else:
                    v = rd_value(t, take)
                    if take:
                        out[key] = v
    except EmissionRefused:
        raise
    except Exception as exc:                                       # noqa: BLE001
        raise EmissionRefused(
            "%s: the metadata could not be walked (%s: %s); nothing was written."
            % (path, type(exc).__name__, exc)) from None
    return out


#: The round-trip tolerance for the fold-back proof.  A MODULE CONSTANT and not a CLI flag, on
#: purpose: a tolerance that can be passed on a command line is a tolerance that can be raised
#: until the guard is off.  :func:`fold_back_tensor` takes it as a keyword that only the probe
#: passes, which is a test-only door and is named as such at the one call site that uses it.
#:
#: The values are not arbitrary.  `A` is the normalized Sylvester transform, whose scale is
#: exactly 1/32 at block 1024, and `A^-1 A` re-applies the same 20 add/subtract stages, so the
#: round trip differs from the identity only by fp32 rounding of a sum of 1024 terms.  Measured
#: on the real decoded `blk.0.ssm_out.weight` the absolute delta is 0.0 (the values are ternary
#: times a per-128 scale, so the dynamic range is narrow); the constants below leave four orders
#: of magnitude of headroom for a tensor with a wide one.
FOLD_ROUNDTRIP_RTOL = 1e-4
FOLD_ROUNDTRIP_ATOL = 1e-6

#: How much of the ternary lattice the unfold must destroy, as a fraction of elements.  PTQ1_0
#: stores three values per 128-wide block, so a tensor in `A`'s basis sits ON that lattice and a
#: tensor in the primal basis does not -- the rotation is a dense mix, and it is the whole reason
#: the fold is applied before quantisation rather than after.  Measured on the real file
#: (dl/ternruntime3/10_probe_fold.txt, P6): the stored `blk.0.ssm_out.weight` scores 1.000000 on
#: the lattice and the unfolded tensor scores 0.000000, so this threshold has the entire range as
#: margin.
#:
#: WHY A SECOND GUARD IS NEEDED AT ALL.  The round-trip check proves that `A^-1` and `A` are an
#: exact inverse pair -- it is a check on the ARITHMETIC.  It is structurally blind to being
#: applied to the wrong starting point, and measured blind to the KEY: `A^-1 A = I` holds for
#: any sign vector, so a corrupted sign table round-trips perfectly.  This second check is a
#: check on the BYTES: if they were never on the lattice, they were never folded, and the
#: declaration that says otherwise is wrong.  Two guards that can fail independently.
FOLD_MIN_DENSIFICATION = 0.30


# --------------------------------------------------------------------------- the fold-back
def _roundtrip_tol(fp32: np.ndarray, rtol=None, atol=None) -> float:
    r = FOLD_ROUNDTRIP_RTOL if rtol is None else float(rtol)
    a = FOLD_ROUNDTRIP_ATOL if atol is None else float(atol)
    scale = float(np.abs(fp32).max()) if fp32.size else 0.0
    return a + r * scale


def ternary_fraction(w, block=128, tol=1e-3):
    """The fraction of elements sitting on the ternary lattice ``PTQ1_0`` stores: for each
    128-wide block, within ``tol`` of the block's own scale, of 0 or of ``+-d``.

    Exposed because the guard that uses it is only as good as the number it computes, and a
    caller that wants to know WHY a tensor was refused should be able to recompute it.
    """
    a = np.asarray(w, dtype=np.float64).reshape(-1, block)
    d = np.abs(a).max(axis=1, keepdims=True)
    d = np.where(d > 0, d, 1.0)
    r = np.abs(a) / d
    return float(((r < tol) | (np.abs(r - 1.0) < tol)).mean())


def fold_back_tensor(name, fp32, width, block, signs, *, rtol=None, atol=None,
                     min_densification=None):
    """``A^-1``, and NOTHING ELSE: the unfold, with the file's head order left exactly as it is.

    Returns ``(primal, max_delta)`` where ``max_delta`` is the worst round-trip residual of the
    unfold (``max |A(A^-1 W) - W|``).  Raises :class:`EmissionRefused` when the residual exceeds
    :data:`FOLD_ROUNDTRIP_RTOL`-scaled tolerance, i.e. this function does not RETURN a tensor it
    has not checked.

    ``fp32`` is the decoded tensor in ggml's flat layout: ``ne[0]`` (the rotation axis, the
    contraction axis of the matmul) is the LAST axis, and every 1024-block is a contiguous run.
    ``signs`` is the contract's own vector for ``width``; the same vector serves both directions
    (``A^-1 = D H`` is the same signs in the opposite ORDER), so no second sign table is needed.

    THE HEAD ORDER IS NOT THIS FUNCTION'S BUSINESS, and the correction this file carries is that it
    used to think it was.  ``fold_back_tensor`` is handed ONE ROW of ``width`` and transforms it;
    the v axis of every GDN tensor but ``ssm_out`` is the tensor's OUTER axis, which a row-local
    function cannot even see.  Re-indexing is therefore :func:`gdn_v_axis_permutation`'s job, at the
    layer that knows the tensor's whole shape -- and for ``ssm_out`` the answer is "do not re-index
    it at all", because the fold this file is unfolding kept that one tensor GROUPED.

    The removed code re-indexed ``ssm_out`` by ``argsort(pi)``, i.e. AWAY from grouped and toward
    tiled, and carried a guard that could not see the direction: it built ``a_tiled``,
    ``a_grouped = a_tiled[pi]`` and verified ``primal @ a_grouped == tiled @ a_tiled``, an identity
    that holds in EITHER orientation by construction (both sides are the same tensor read two ways).
    A green guard over a reversed tensor is the failure shape this file exists to prevent, so the
    guard is gone with the code it guarded, and the direction is now measured against the consumer's
    own bytes instead -- dl/ssmdir/logs/witness2.txt (``1.000000`` unpermuted, ``0.361229`` for
    ``argsort(pi)``, ``0.333399`` random) and dl/ssmdir/logs/witness4.txt.
    """
    arr = np.asarray(fp32, dtype=np.float32)
    if arr.size % width:
        raise EmissionRefused(
            "%s: %d elements is not a multiple of the declared sign width %d; the fold set and "
            "the tensor table disagree and nothing was written."
            % (name, arr.size, width))
    rows = arr.reshape(-1, width)

    # --- the unfold: A^-1 = D H, signs AFTER the transform
    primal = H.apply_inverse(rows, signs, block, axis=-1)

    # --- GUARD 1, on the ARITHMETIC: A(A^-1 W) must reproduce the stored bytes to fp32 roundoff
    back = H.apply_forward(primal, signs, block, axis=-1)
    delta = float(np.abs(back - rows).max())
    tol = _roundtrip_tol(rows, rtol, atol)
    if not np.isfinite(delta) or delta > tol:
        raise EmissionRefused(
            "the fold-back round trip for %s did not verify: max|A(A^-1 W) - W| = %.6e exceeds "
            "the tolerance %.6e.  An unfold that does not round-trip is not an unfold, and a "
            "checkpoint written from it would be silently wrong.  Nothing was written."
            % (name, delta, tol))

    # --- GUARD 2, on the BYTES: the unfold must DENSIFY the tensor.
    # This one is not redundant with guard 1, and the difference is measured rather than argued
    # (dl/ternruntime3/10_probe_fold.txt, P6.6/P6.7): the round trip is EXACT on this file
    # (delta = 0.0e+00), so its tolerance can never fire; and `A^-1 A = I` holds for ANY sign
    # vector, so the round trip cannot see a corrupted key either.  What it also cannot see is
    # being applied to the wrong STARTING POINT -- an already-primal tensor round-trips perfectly
    # through `A^-1` and `A` and comes out unchanged.  The lattice can see it: PTQ1_0 stores three
    # values per 128-block, so a tensor in `A`'s basis sits on that lattice and a primal one does
    # not.  If the unfold leaves the tensor on the lattice, the bytes were never folded and the
    # declaration that says they were is wrong.
    dens = FOLD_MIN_DENSIFICATION if min_densification is None else float(min_densification)
    tf_before = ternary_fraction(rows)
    tf_after = ternary_fraction(primal)
    if not (tf_before - tf_after >= dens):
        raise EmissionRefused(
            "the unfold of %s did NOT densify the tensor: the ternary-lattice fraction went from "
            "%.6f to %.6f, a drop of %.6f against the required %.6f.  Bytes that stay on the "
            "lattice after being rotated out of the rotation's basis were never IN that basis, so "
            "the file's declaration that this tensor is folded is wrong -- and an unfold applied "
            "to an already-primal tensor is an extra application, which is a silently wrong "
            "weight.  Nothing was written."
            % (name, tf_before, tf_after, tf_before - tf_after, dens))

    return np.ascontiguousarray(primal, dtype=np.float32), delta


# ------------------------------------------------------- the FILE's own head order and A space

#: The stems whose v axis the FILE keeps in the packed (tiled) head order, so that the canonical
#: order a consumer reads is ``a_grouped = a_tiled[perm]``.  This list is not a guess: it is
#: ``runtime.py:179-192`` of the MLX runtime shipped with the reference pack, branch for branch,
#: and the one stem that is NOT here is the one that matters --
#:
#:     def reorder(a, stem):
#:         if nv == nk:
#:             return a
#:         if stem == "attn_qkv.weight":
#:             qk = 2 * nk * hk
#:             return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
#:         if stem == "attn_gate.weight":
#:             return a[vperm(hd)]
#:         if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
#:             return a[vperm(1)]
#:         if stem == "ssm_conv1d.weight":
#:             qk = 2 * nk * hk
#:             return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
#:         return a
#:
#: ``ssm_out.weight`` is absent from every branch, i.e. the reference uses the FILE's own order for
#: it -- and that file is the folded one, whose ``prism.hadamard.gdn_v_grouped = 1`` says the fold
#: kept it GROUPED.  MEASURED on the reference's own bytes (dl/ssmdir/logs/witness2.txt): the pack's
#: layer-0 ``linear_attn.out_proj`` is ternary-identical to the folded GGUF's ``blk.0.ssm_out.weight``
#: as stored, ``1.000000``, and only ``0.361229`` after ``argsort(pi)`` (random control 0.333399).
#: The other four branches are the same statement about the other five stems, and two of them are
#: byte-measured too (``in_proj_z``: ``1.000000`` for ``rows[pi]`` against ``0.361847`` for
#: identity, witness4 A; ``A_log``: ``4.910970e-08`` against ``log(-ssm_a[vperm])``, witness D3).
_GDN_V_TILED_STEMS = ("attn_gate.weight", "attn_qkv.weight", "ssm_conv1d.weight",
                      "ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias")

#: Elements above which a v-axis re-indexing is REFUSED rather than attempted.  The permutation is a
#: gather on the tensor's outer axis, which no row-chunked reader can do chunk by chunk, so such a
#: tensor is materialised whole.  The released 27B's largest is ``blk.*.attn_qkv.weight`` at
#: 10,240 x 5,120 = 52,428,800 elements (14 MB of PQ2_0 -> 210 MB of fp32); the bound is one
#: doubling above that, so a future file cannot silently turn this into a swap storm.
GDN_PERMUTE_MAX_ELEMS = 1 << 26


def gdn_v_axis_geometry(kv) -> tuple | None:
    """``(n_v, n_k, v_width, hk)`` from the file's own GDN declaration, or ``None`` if unreadable.

    Exactly the four numbers ``runtime.py:111-115`` reads, under this tree's metadata spellings:
    ``time_step_rank`` = n_v (value heads), ``group_count`` = n_k (q/k heads), ``inner_size`` =
    n_v * hd (the v-axis width), ``state_size`` = hk (the key head dim, which sets the qk prefix
    that ``attn_qkv`` and ``conv1d`` carry ahead of their v segment).
    """
    try:
        n_v = int(kv.get("qwen35.ssm.time_step_rank") or 0)
        n_k = int(kv.get("qwen35.ssm.group_count") or 0)
        v_width = int(kv.get("qwen35.ssm.inner_size") or 0)
        hk = int(kv.get("qwen35.ssm.state_size") or 0)
    except (TypeError, ValueError):
        return None
    if n_v <= 0 or n_k <= 0 or v_width <= 0 or hk <= 0:
        return None
    return (n_v, n_k, v_width, hk)


def gdn_v_axis_permutation(name, dims, geom) -> tuple | None:
    """``(row_offset, length, stride, perm)`` -- the row re-indexing ``name`` owes, or ``None``.

    ``perm[j]`` is the SOURCE row of output row ``j``, so ``out[off + j] = src[off + perm[j]]`` for
    ``j`` in ``[0, length)`` and every row is ``stride`` elements wide.  ``None`` means "the file
    already carries the canonical order", which is true for ``ssm_out.weight`` BY DECLARATION
    (``gdn_v_grouped``) and for every tensor outside the v-axis family.

    WHY THIS LIVES OUTSIDE ``fold_back_tensor``.  ``fold_back_tensor`` is handed ONE ROW of
    ``width = dims[0]`` -- the rotation axis, ggml's fastest-varying one -- and the rotation is a
    per-row operation on that axis.  The v axis of every stem in :data:`_GDN_V_TILED_STEMS` is the
    OTHER one: ``attn_gate`` is (5120, 6144) as stored and its 6144 is the OUTER axis, so a function
    that only ever sees a row of 5120 cannot re-index it.  The unfold and this gather commute (they
    act on different axes), which is why applying it afterwards is the same operation.

    The geometry comes from the file itself and every number is CHECKED against the tensor's own
    shape rather than assumed: a stem that matches while its shape does not is refused by name,
    because the alternative is re-indexing the wrong axis of a tensor nobody will look at again.
    """
    stem = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if stem not in _GDN_V_TILED_STEMS:
        return None
    if geom is None:
        raise EmissionRefused(
            "%s carries the GDN v axis, whose head order is a property of the FILE, but the file's "
            "own GDN geometry (qwen35.ssm.time_step_rank / group_count / inner_size / state_size) "
            "is not readable, so the re-indexing it owes cannot be computed.  Guessing the head "
            "order here is exactly the mistake this tool is being corrected for.  Nothing was "
            "written." % name)
    n_v, n_k, v_width, hk = geom
    if n_v == n_k:
        return None                      # nothing is grouped; the reference's reorder is the identity

    nelem = 1
    for d in dims:
        nelem *= int(d)
    if stem in ("ssm_a", "ssm_dt.bias"):
        # 1-D, one value per value head: the head axis IS the flat axis.
        stride, want, off_rows = 1, n_v, 0
    elif stem in ("ssm_alpha.weight", "ssm_beta.weight"):
        stride, want, off_rows = int(dims[0]), n_v, 0
    elif stem == "attn_gate.weight":
        stride, want, off_rows = int(dims[0]), v_width, 0
    else:                                 # attn_qkv.weight, ssm_conv1d.weight
        stride, want, off_rows = int(dims[0]), v_width, 2 * n_k * hk
    if stride <= 0 or nelem % stride:
        raise EmissionRefused(
            "%s: %d elements is not a whole number of %d-wide rows, so its v axis cannot be "
            "re-indexed.  Nothing was written." % (name, nelem, stride))
    rows = nelem // stride
    if rows != off_rows + want:
        raise EmissionRefused(
            "%s as stored is %d rows of %d, but the GDN geometry (n_v=%d n_k=%d v_width=%d "
            "state_size=%d) says its v axis is rows [%d, %d) -- %d qk rows then %d v rows.  A "
            "re-indexing applied to a shape that is not this one would corrupt a tensor silently, "
            "so it is refused by name.  Nothing was written."
            % (name, rows, stride, n_v, n_k, v_width, hk, off_rows, off_rows + want,
               off_rows, want))
    perm = H.gdn_v_permutation(want, n_v, n_k)
    if sorted(int(v) for v in perm) != list(range(want)):
        raise EmissionRefused(
            "the GDN v permutation for width=%d n_v=%d n_k=%d is not a permutation of 0..%d; "
            "refusing to re-index %s with it.  Nothing was written."
            % (want, n_v, n_k, want - 1, name))
    return (off_rows, want, stride, perm)


def apply_gdn_v_axis_permutation(flat, spec, name):
    """Re-index the rows of ``flat`` by ``spec`` as returned by :func:`gdn_v_axis_permutation`."""
    off_rows, length, stride, perm = spec
    arr = np.ascontiguousarray(flat, dtype=np.float32)
    if arr.size % stride:
        raise EmissionRefused(
            "%s: %d elements is not a whole number of %d-wide rows.  Nothing was written."
            % (name, arr.size, stride))
    rows = arr.size // stride
    if off_rows + length > rows:
        raise EmissionRefused(
            "%s: the v-axis re-indexing wants rows [%d, %d) of %d.  Nothing was written."
            % (name, off_rows, off_rows + length, rows))
    order = np.arange(rows, dtype=np.int64)
    order[off_rows:off_rows + length] = off_rows + np.asarray(perm, dtype=np.int64)
    return np.ascontiguousarray(arr.reshape(rows, stride)[order].reshape(-1))


def gdn_a_log_from_stored(name, flat, geom):
    """The LOG-SPACE ``A_log`` the name ``linear_attn.A_log`` denotes, or ``None`` for other names.

    The FILE stores ``ssm_a`` as ``-exp(A_log)``: ``runtime.py:248-251`` reads it, refuses a value
    that is not negative, and derives ``a = np.log(-a)`` before handing it to the model -- i.e. the
    MLX namespace's ``A_log`` is the log-space parameter and the GGUF's is its negated exponential.
    MEASURED both ways on this box:

      * the reference pack's own ``layers.0.linear_attn.A_log`` is ``[-5.5625, -1.078125]`` while
        the folded GGUF's ``blk.0.ssm_a`` is ``[-0.340233, -0.003839]``, and ``log`` of the second
        reproduces the first to ``4.910970e-08`` (dl/ssmdir/logs/witness.txt D3);
      * this engine's own consumer is ``src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu``
        ``g[out_index] = -expf(A_log[row]) * softplus(...)`` -- so an ``a_log`` object carrying the
        file's raw ``-exp(A_log)`` makes the engine compute ``-exp(-exp(A_log))``, a gate of about
        ``-0.99`` where the correct one is about ``-0.005``.

    The emitted object is NAMED ``linear_attn.A_log`` and the 27B text-core recipe casts it through
    verbatim (``qwen3_6_27b/recipe.py:123-124``), so the converter cannot be the place this is
    fixed without changing a recipe that is shared with the HF-sourced, WORKING artifacts, whose
    source really is log space.  It is fixed here, where the file's convention is being translated.

    A non-negative value is refused by name rather than logged into a NaN: ``log(-a)`` has no real
    answer there, and a silent NaN in a gate is a model that runs and says nothing.
    """
    stem = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if stem != "ssm_a":
        return None
    if geom is None:
        raise EmissionRefused(
            "%s is the file's ``-exp(A_log)`` and the emitted object is named ``A_log``, but the "
            "file's GDN geometry is not readable, so this converter cannot tell a GDN ``ssm_a`` "
            "from any other tensor of that name and will not guess the space.  Nothing was "
            "written." % name)
    a = np.asarray(flat, dtype=np.float64)
    if not (a < 0).all():
        raise EmissionRefused(
            "%s holds a non-negative value, so it is not the file's ``-exp(A_log)`` and the "
            "log-space object the consumer reads cannot be derived from it.  The reference runtime "
            "refuses this file too (runtime.py:249, 'Invalid stored SSM A').  Nothing was written."
            % name)
    return np.log(-a).astype(np.float32)


def gdn_head_order_declaration(declared, emitted) -> str:
    """``"grouped"``/``"tiled"``: the order the EMITTED ``ssm_out`` is actually in.

    A single formula so the declaration sites cannot drift.  Both of this converter's routes
    leave/put ``blk.*.ssm_out.weight`` in the fold's GROUPED head order::

        declared folded + fold_back      the fold kept it grouped, and fold_back changes only the
                                         BASIS (``A^-1``), never the head order -> grouped
        declared folded + pass_through   the bytes are copied as they stand, and a folded file is
                                         grouped -> grouped
        declared primal + pass_through   an unrotated file of this same family carries it TILED
                                         (``gguf_hadamard.ssm_out_head_order``) -> tiled

    The pre-correction formula was ``grouped if emitted == BASIS_FOLDED else tiled``, which types
    the ROUTE and not the bytes: it declared ``tiled`` over a ``fold_back`` output whose ssm_out was
    grouped, and the shipped ``config.json`` said so out loud (``prism.fold.basis = primal`` beside
    ``prism.fold.gdn_head_order = tiled``).
    """
    if emitted == FR.BASIS_FOLDED or declared == FR.BASIS_FOLDED:
        return "grouped"
    return "tiled"


# --------------------------------------------------------------------------- plan + write
def plan_tensors_routeaware(kv, tensors, arch, route, declared=None):
    """The tensor plan, and the ROUTE GATE, in one place.

    Mirrors ``gguf_extract.plan_tensors`` field for field, with the one difference that the
    rotated-file refusal is replaced by the route decision.  The decision is made from the
    LEDGER and the DECLARED BASIS, never from the flag alone -- see
    :func:`gguf_fold_route.converter_admission`.
    """
    if arch not in N.RULES:
        raise EmissionRefused(
            "no GGUF tensor-name rules for architecture %r; this tree knows %s."
            % (arch, ", ".join(sorted(N.RULES))))

    # What the SOURCE declares.  Read by the decision module, which `gguf_extract.py` also
    # calls, so the two tools cannot drift about which files count as folded.  It is also what
    # `main` derives the ROUTE from when the operator names none, so it is passed in rather than
    # recomputed: one read, one answer.
    if declared is None:
        declared = FR.declared_basis_of(kv)
    if declared == FR.BASIS_FOLDED:
        # The head-order question, which is a separate one from the declaration: a file that
        # declares a rotation and not which head order it kept raises UnprovenRotation here, and
        # one that declares the fold and says the grouped order was NOT kept is refused below.
        try:
            grouped = H.gdn_ssm_out_is_grouped(kv)
        except H.UnprovenRotation as exc:
            raise EmissionRefused("%s  Nothing was written." % exc) from None
        if not grouped:
            raise EmissionRefused(
                "the file declares a Hadamard fold (%sversion=%r) but says "
                "%sgdn_v_grouped is False.  A folded file keeps blk.*.ssm_out.weight in the "
                "fold's GROUPED head order -- that is what the field means -- so a False here "
                "describes a file this converter has never seen and will not guess at.  "
                "Nothing was written."
                % (H.PREFIX, kv.get(H.PREFIX + "version"), H.PREFIX))

    if declared == FR.BASIS_FOLDED:
        contract = H.contract(kv)
        if contract is None:
            raise EmissionRefused("a rotated file whose contract does not parse.  Nothing was written.")
        fwd = list(contract["forward_names"])
        inv = list(contract["inverse_names"])
        by_width = contract["by_width"]
        block = int(contract["block_size"])
    else:
        contract, fwd, inv, by_width, block = None, [], [], {}, 0

    # The declared basis of the OUTPUT, which is what the ledger must agree with.
    #
    # fold_back applies A^-1, so the emitted weights are primal.  pass_through applies nothing, so
    # the emitted weights are in the SAME basis as the source's -- `primal` for a primal source,
    # `folded` for a rotated one (that is Route A: the runtime owes the transform).  Written the
    # other way round (`folded` whenever the route is pass_through) it declared `folded` over
    # unrotated bytes; the post-condition caught that, correctly, but only after the whole file
    # had been written.  MEASURED 2026-09-20: with the route now derived from the source,
    # `gguf_fold_back.py --src Bonsai-27B-Q1_0.gguf --out <dir>` takes `pass_through`, so the
    # wrong formula would have spent the entire 53.79 GB emission before refusing it.
    emitted = FR.BASIS_PRIMAL if route == FR.ROUTE_FOLD_BACK else declared

    plan, seen = [], {}
    for name, dims, ttype, off in tensors:
        if ttype not in K.DEQUANTIZERS:
            raise EmissionRefused(
                "tensor %r is stored as ggml type %d (%s) and this reader decodes %s.  "
                "Nothing was written." % (name, ttype, K.type_name(ttype), K.SUPPORTED_NAMES))
        got = N.translate(arch, name, dims, kv)
        if got is None:
            raise EmissionRefused(
                "tensor %r has no name rule for architecture %r.  Nothing was written."
                % (name, arch))
        key, shape = got
        if key in seen:
            raise EmissionRefused(
                "two GGUF tensors map to the same HF key %r: %s and %s"
                % (key, seen[key], name))
        seen[key] = name
        nelem = 1
        for d in dims:
            nelem *= int(d)
        if int(np.prod(shape)) != nelem:
            raise EmissionRefused("shape reversal changed the element count for %r" % (name,))
        plan.append((name, key, tuple(shape), ttype, off, nelem, tuple(int(d) for d in dims)))
    return plan, declared, emitted, fwd, inv, by_width, block, contract


def basis_declaration(route, declared, emitted) -> dict:
    """The basis facts about the emitted checkpoint, typed, for ``config.json``.

    The same facts the safetensors ``__metadata__`` block carries, under the same key spellings,
    computed from the same three inputs by the same module that writes that block -- so a
    consumer that reads the config and one that reads the header cannot disagree about which
    basis the weights are in.

    ``emitted`` is not a label for the route, it is a statement about the BYTES: ``primal``
    exactly when this run applied ``A^-1`` (route ``fold_back``), and otherwise whatever the
    source declared, because ``pass_through`` copies the source's weights without transforming
    them.  ``prism.fold.runtime_must_transform`` is true in exactly that second case.
    """
    return {
        FR.BASIS_KEY: emitted,
        "prism.fold.route": route,
        "prism.fold.source.declared": declared,
        "prism.fold.gdn_head_order": gdn_head_order_declaration(declared, emitted),
        "prism.fold.runtime_must_transform": emitted == FR.BASIS_FOLDED,
    }


def write_safetensors_routeaware(dest, src, data_offset, plan, route, declared, emitted, fwd, inv,
                                 by_width, block, kv, ledger, *, rtol=None, atol=None,
                                 bounded_by=None):
    """Stream every tensor to ``dest``, applying the route's one transform on the way.

    The header's ``__metadata__`` carries the basis.  It is written BEFORE any payload byte, so a
    reader that finds the checkpoint finds the declaration that says what its bytes are.
    """
    fold_names = set(fwd) | set(inv)
    # The file's own GDN geometry, when this run OWES the file's conventions to the emitted bytes.
    # Only `fold_back` on a `folded` source does: Route A copies the file as it stands and the
    # runtime owes every transform.
    gdn_geom = (gdn_v_axis_geometry(kv)
                if (declared == FR.BASIS_FOLDED and route == FR.ROUTE_FOLD_BACK) else None)

    header, offset = {}, 0
    for _name, key, shape, _ttype, _off, nelem, _dims in plan:
        nbytes = nelem * 2
        header[key] = {"dtype": "BF16", "shape": list(shape),
                       "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = {
        FR.BASIS_KEY: emitted,
        "prism.fold.route": route,
        "prism.fold.source.declared": declared,
        "prism.fold.gdn_head_order": gdn_head_order_declaration(declared, emitted),
        "prism.fold.transform": str(kv.get(H.PREFIX + "transform", "")),
        "prism.fold.block_size": str(kv.get(H.PREFIX + "block_size", "")),
        "prism.fold.runtime_must_transform": "true" if emitted == FR.BASIS_FOLDED else "false",
        # A bounded diagnostic run says so, in the file.  Nothing downstream has to know this
        # flag exists to be safe, because the DEFAULT is "false" and only `--only` sets it -- but
        # a consumer that wants a complete checkpoint can check it rather than guess from a
        # tensor count.
        "prism.fold.bounded": "true" if bounded_by else "false",
        "prism.fold.bounded_by": str(bounded_by or ""),
        "prism.fold.tensors_written": str(len(plan)),
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((-(8 + len(blob))) % 8)
    written = 0
    with open(src, "rb") as fh, open(dest, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        # Flushed before the first payload byte.  MEASURED 2026-09-20: this box's Python has
        # io.DEFAULT_BUFFER_SIZE = 131072 (dl/foldroute/logs/**, s11) and the header for the
        # ternary's 851-tensor plan is 113,488 B, so WITHOUT this call nothing of the declaration
        # reaches the file until the first tensor's payload write completes -- and on this box the
        # first planned tensor (output.weight, 1,271,398,400 elements) does not complete at all in
        # 11.5 minutes of CPU (measured by the peer line: 0 bytes emitted, stationary thrash).
        # A killed run then leaves a 0-byte .partial and the declaration of what its bytes are
        # dies with the process.  The header is written first precisely so that a reader finds it;
        # this makes that true of the FILE and not only of the buffer.
        out.flush()
        for name, key, _shape, ttype, off, nelem, dims in plan:
            need = K.tensor_nbytes(ttype, nelem)
            fh.seek(data_offset + off)
            raw = fh.read(need)
            if len(raw) != need:
                raise EmissionRefused(
                    "tensor %s needs %d bytes at offset %d but the file ends after %d"
                    % (name, need, off, len(raw)))
            fp32 = K.to_fp32(ttype, raw, nelem)
            if not np.isfinite(fp32).all():
                raise EmissionRefused("tensor %s decoded to non-finite values" % name)
            if declared == FR.BASIS_FOLDED and name in fold_names:
                width = int(dims[0])                      # ggml ne[0] is the rotation axis
                signs = by_width.get(width)
                if signs is None:
                    raise EmissionRefused(
                        "tensor %s has ggml ne[0]=%d and the contract declares no sign vector of "
                        "that width (declared: %s).  A folded tensor whose width is not declared "
                        "cannot be unfolded, and rounding it through as if it were primal would "
                        "be the silent wrong.  Nothing was written."
                        % (name, width, sorted(by_width)))
                if route == FR.ROUTE_FOLD_BACK:
                    fp32, delta = fold_back_tensor(name, fp32, width, block, signs,
                                                   rtol=rtol, atol=atol)
                    ledger.record(name, True, True, delta)
                else:
                    # pass_through: NOTHING is unfolded, and the ledger must show it.  The
                    # record is written so a caller can see the tensor was considered and left
                    # alone, which is a different statement from "was not seen".
                    ledger.record(name, False, True, 0.0)
            # The FILE's own conventions, restored here, because here is where the whole shape
            # is known.  `None` from either means the file already carries the canonical form.
            gperm = gdn_v_axis_permutation(name, dims, gdn_geom)
            if gperm is not None:
                fp32 = apply_gdn_v_axis_permutation(fp32, gperm, name)
            a_log = gdn_a_log_from_stored(name, fp32, gdn_geom)
            if a_log is not None:
                fp32 = a_log
            # The engine's rmsnorm epilogue adds one to this weight
            # (src/ops/kernel/rmsnorm.cuh:22) and the GDN control projection goes
            # through that epilogue with the flag set
            # (src/ops/generic/rowsplit_generic.cu:519), so the artifact must hold
            # `gamma - 1`.  A GGUF holds the plain `gamma`.  Set + why:
            # tools/convert/unit_offset_norms.py
            fp32 = unit_offset_norms.stored_values(fp32, key)
            out.write(K.fp32_to_bf16_bytes(fp32))
            written += 1
        out.flush()
    return written


# --------------------------------------------------------------------------- the entry point
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="input .gguf")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--fold-route", default=None, choices=list(FR.ROUTE_VALUES),
                    help="which of the two routes to take.  DEFAULT: derived from the source "
                         "itself -- a source that declares a fold gets 'fold_back', one that "
                         "declares none gets 'pass_through'.  Give the flag to override; "
                         "'refuse' emits nothing and is never chosen for you.")
    ap.add_argument("--report", action="store_true", help="print the route decision and ledger")
    ap.add_argument("--only", default=None, metavar="REGEX",
                    help="DIAGNOSTIC: emit only the tensors whose ggml name matches REGEX.  The "
                         "route gate is then asked about the SUBSET this run actually processes "
                         "(a partial traverse cannot satisfy a whole-file predicate, and pretending "
                         "otherwise would be the weakening this gate exists to prevent), and the "
                         "emitted checkpoint DECLARES ITSELF PARTIAL under prism.fold.bounded so no "
                         "consumer can mistake it for a complete checkpoint.  Full-file correctness "
                         "is NOT claimed for a bounded run.")
    ap.add_argument("--simulate-partial-fold", action="store_true",
                    help="TEST-ONLY RED CONTROL: make the post-condition ask about the WHOLE "
                         "declared fold set while walking a bounded subset, so the gate is shown "
                         "to fire and to leave ZERO files.  It cannot admit anything; it can only "
                         "make the gate stricter.")
    args = ap.parse_args(argv)

    # The route is decided BELOW, once the source has been read: a source that declares its
    # basis has already said which of the two routes is its own, and `refuse` -- "emit
    # nothing" -- is a value an operator types, never the answer to "the operator said
    # nothing".  An explicit --fold-route overrides the derivation.
    #
    # A source this reader cannot open is THIS tool's refusal, not a traceback.
    try:
        kv, tensors, data_offset = K.read_tensor_table(args.src)
    except (X.ExtractionRefused, N.UnmappedTensors):
        raise
    except Exception as exc:                                   # noqa: BLE001
        raise EmissionRefused(
            "%s cannot be read as a GGUF tensor table: %s: %s  Nothing was written."
            % (args.src, type(exc).__name__, exc)) from None
    arch = str(kv.get("general.architecture") or "")

    # The contract's arrays are above the repo reader's summary threshold, so the sign VALUES
    # are not in `kv` and the unfold cannot be done from it.  Read them, and only them, from the
    # file itself.  Everything else stays as `read_tensor_table` produced it.
    wanted = [H.PREFIX + suffix for suffix in
              ("version", "block_size", "transform", "axis", "sign_mode", "sign_widths",
               "sign_values", "weight_names", "inverse_weight_names", "gdn_v_grouped")]
    wanted += ["qwen35.ssm.time_step_rank", "qwen35.ssm.group_count", "qwen35.ssm.inner_size",
               "qwen35.ssm.state_size", "general.architecture"]
    real = read_metadata_values(args.src, wanted)
    kv = dict(kv)
    kv.update(real)
    if H.PREFIX + "version" in real:
        for key in wanted[:10]:
            if key not in real:
                raise EmissionRefused(
                    "the file declares %sversion=%r but its metadata has no %r; the contract "
                    "cannot be parsed and nothing was written."
                    % (H.PREFIX, real[H.PREFIX + "version"], key))

    # The DERIVATION, and the one place this file decides a route: the source's own
    # declaration names it when the operator names none (`general.basename == 'folded'` or a
    # `prism.hadamard.*` block -> fold_back; neither -> pass_through).  No input of that
    # function is `refuse`.
    declared = FR.declared_basis_of(kv)
    if args.fold_route is not None:
        route = args.fold_route
    else:
        route = FR.route_for_declared_basis(declared)

    try:
        rep = N.coverage(kv, tensors, arch)
    except (X.ExtractionRefused, N.UnmappedTensors):
        raise
    except Exception as exc:                                   # noqa: BLE001
        raise EmissionRefused("%s: the tensor table was not name-mapped: %s: %s"
                              % (args.src, type(exc).__name__, exc)) from None
    N.require_full_coverage(rep)

    plan, declared, emitted, fwd, inv, by_width, block, contract = plan_tensors_routeaware(
        kv, tensors, arch, route, declared)

    bounded_by = None
    if args.only:
        import re
        try:
            rx = re.compile(args.only)
        except re.error as exc:
            raise EmissionRefused("--only %r is not a regular expression: %s.  Nothing was written."
                                  % (args.only, exc)) from None
        before = len(plan)
        plan = [p for p in plan if rx.search(p[0])]
        if not plan:
            raise EmissionRefused(
                "--only %r matched none of the %d planned tensors; a bounded run that bounds "
                "itself to nothing is a mistake, not a result.  Nothing was written."
                % (args.only, before))
        bounded_by = args.only
        print("*** BOUNDED DIAGNOSTIC RUN: --only %r selects %d of %d tensors. ***"
              % (args.only, len(plan), before))
        print("*** The emitted checkpoint declares prism.fold.bounded=true. ***")

    fold_set = sorted(set(fwd) | set(inv))
    if bounded_by is not None:
        # Ask the gate about the subset this run actually processes.  The plan gives the GGUF
        # names; the fold set gives the ones that need the transform, so the intersection is the
        # predicted expected-set.
        fold_set = [n for n in fold_set if any(p[0] == n for p in plan)]
    if args.simulate_partial_fold:
        # TEST-ONLY RED CONTROL, and named as such.  It makes the ledger ask about the WHOLE
        # declared fold set even though the run only walks a bounded subset, which is exactly the
        # shape of a partial fold-back -- so the post-condition must refuse it.  There is no value
        # of this flag that admits anything; it can only make the gate stricter.
        fold_set = sorted(set(fwd) | set(inv))
        print("*** RED CONTROL: --simulate-partial-fold asks the gate about all %d declared "
              "tensors while walking %d.  The post-condition MUST refuse. ***"
              % (len(fold_set), len(plan)))
    ledger = FR.FoldLedger(expected_names=fold_set)

    # The EARLY half of the gate: a property of the request and the declarations, knowable now.
    # It is deliberately not the whole gate -- see `gguf_fold_route.route_precheck` for why, and
    # why the split is not a loosening.  It runs before the output directory is touched.
    try:
        FR.route_precheck(route, declared, emitted)
    except FR.FoldRouteRefused as exc:
        raise EmissionRefused("%s  Nothing was written." % exc) from None

    print("source  : %s" % args.src)
    print("arch    : %s  layers %d main + %d draft" % (arch, rep["main_layers"], rep["nextn_layers"]))
    print("declared: %s   route: %s   emitted declaration: %s%s"
          % (declared, route, emitted,
             "  (route derived from the source's own declaration; no --fold-route given)"
             if args.fold_route is None else
             "  (--fold-route=%s was supplied on this command line)" % route))
    print("fold set: %d forward + %d inverse = %d" % (len(fwd), len(inv), len(fwd) + len(inv)))
    print("precheck: admitted (the request and the declarations agree); the post-condition runs "
          "inside the output transaction and reads the ledger")

    out_dir = Path(args.out)
    created_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    have_head = any(n == "output.weight" for n, _d, _t, _o in tensors)
    # THE OUTPUT TRANSACTION, and the POST-CONDITION INSIDE IT.
    #
    # The post-condition is checked against the ledger the write loop actually filled, and it is
    # checked INSIDE the `with` -- which is the whole point.  `OutputTransaction.__exit__`
    # discards the staged files when the body raises, and publishes them with `os.replace` only
    # when it does not.  Putting the check after the block (as the first version of this file
    # did) meant a refusal fired on a checkpoint that was ALREADY PUBLISHED: the tool printed
    # "Nothing was written" over a directory containing a complete model.safetensors.  That is
    # the exact defect this tree's transaction exists to make impossible, and it was found by
    # this gate firing.
    with X.OutputTransaction(out_dir, created_dir) as txn:
        written = write_safetensors_routeaware(
            txn.staging("model.safetensors"), args.src, data_offset, plan, route, declared,
            emitted, fwd, inv, by_width, block, kv, ledger, bounded_by=bounded_by)
        txn.staging("config.json").write_text(
            json.dumps(X.config_from_metadata(
                kv, arch, have_head,
                fold_declaration=basis_declaration(route, declared, emitted)),
                ensure_ascii=False, indent=2)
            + "\n", encoding="utf-8")
        try:
            FR.converter_admission(route, declared, ledger, emitted)
        except FR.FoldRouteRefused as exc:
            # Raising from inside the block makes __exit__ clean up, so this refusal leaves the
            # output directory exactly as it found it.
            raise EmissionRefused("%s  Nothing was written.  %s"
                                  % (exc, ledger.render())) from None
    if args.report:
        print(ledger.render())

    print("extracted %d/%d tensors -> %s" % (written, len(plan), out_dir))
    print("basis   : %s=%s  runtime_must_transform=%s"
          % (FR.BASIS_KEY, emitted, emitted == FR.BASIS_FOLDED))
    return 0


if __name__ == "__main__":
    # Every way this tool can decline is NAMED.  The contract's own two refusals are listed
    # because they are reachable from here and would otherwise surface as a traceback with a
    # non-zero exit -- which reads to a caller as a crash rather than as a considered refusal,
    # and the difference matters when the alternative is a silently wrong checkpoint.
    try:
        raise SystemExit(main())
    except (EmissionRefused, X.ExtractionRefused, N.UnmappedTensors, FR.FoldRouteRefused,
            H.UnreadableMetadata, H.UnprovenRotation, ValueError) as exc:
        # The inner layers each append the sentence as they wrap, so it can arrive two or three
        # times over.  Say it once: a refusal message that repeats itself reads like three
        # different failures, and this one is meant to be read.
        text = str(exc)
        if "Nothing was written." not in text:
            text = text + "  Nothing was written."
        print("REFUSED: %s" % text, file=sys.stderr)
        raise SystemExit(3)
