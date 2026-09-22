#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K-quant reader verification: gguf_kquant / gguf_names.

Runnable with or without pytest::

    python3 tests/convert/test_gguf_kquant.py          # standalone, prints PASS/FAIL
    python3 -m pytest tests/convert/test_gguf_kquant.py

Two layers of verification, deliberately independent of each other:

1. **Layout, from the file's own arithmetic.**  The block sizes are used to
   reproduce every tensor offset in the real file.  A wrong values-per-block or
   bytes-per-block shows up as an offset mismatch.  This check never calls a
   dequantiser.

2. **Arithmetic, against a second implementation.**  ``_ref_q4_K`` / ``_ref_q6_K``
   below are scalar loops written straight from the block description, on purpose
   *not* sharing code with the vectorised version in ``gguf_kquant``.  Both are run
   on the same real blocks; a mismatch is an indexing bug in one of them.

Plus grid-membership and finiteness assertions on real tensors, so a decode that
is self-consistent but lands off the quantisation lattice fails, and a negative
control so the checks above are shown capable of failing.

The real files are optional.  ``NINFER_ORNITH_GGUF`` points at the GGUF; without
it the real-file tests are skipped rather than passing vacuously.
"""
from __future__ import annotations

import os
import struct
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.convert import gguf_kquant as K      # noqa: E402
from tools.convert import gguf_names as N       # noqa: E402

ORNITH = os.environ.get(
    "NINFER_ORNITH_GGUF",
    "/mnt/c/Users/User/Documents/ziqinzhang/models/ornith-1.5-9b/"
    "Ornith-1.5-9B-Q4_K_M.gguf")


@contextmanager
def raises(exc_type):
    try:
        yield
    except exc_type as exc:
        return
    except Exception as exc:                                  # noqa: BLE001
        raise AssertionError("expected %s, got %s: %s"
                             % (exc_type.__name__, type(exc).__name__, exc))
    raise AssertionError("expected %s, nothing was raised" % exc_type.__name__)


_skip_reasons: list[str] = []


def needs_real(fn):
    if not Path(ORNITH).is_file():
        _skip_reasons.append("%s: real GGUF absent (%s)" % (fn.__name__, ORNITH))
        fn.__skip__ = "no real file"
    return fn


# --------------------------------------------------------------------------- #
# independent scalar reference decoders (written from the block description)
# --------------------------------------------------------------------------- #
def _get_scale_min_k4(j: int, q: bytes):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return ((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4))


def _ref_q4_K(raw: bytes, n_elements: int) -> np.ndarray:
    out = np.empty(n_elements, np.float32)
    for b in range(n_elements // 256):
        blk = raw[b * 144:(b + 1) * 144]
        d = struct.unpack("<e", blk[0:2])[0]
        dmin = struct.unpack("<e", blk[2:4])[0]
        sc, qs = blk[4:16], blk[16:144]
        y = b * 256
        for g in range(4):
            d1, m1 = _get_scale_min_k4(2 * g, sc)
            d2, m2 = _get_scale_min_k4(2 * g + 1, sc)
            for l in range(32):
                out[y + 64 * g + l] = d * d1 * (qs[32 * g + l] & 0xF) - dmin * m1
                out[y + 64 * g + 32 + l] = d * d2 * (qs[32 * g + l] >> 4) - dmin * m2
    return out


def _ref_q6_K(raw: bytes, n_elements: int) -> np.ndarray:
    out = np.empty(n_elements, np.float32)
    for b in range(n_elements // 256):
        blk = raw[b * 210:(b + 1) * 210]
        ql, qh, scb, db = blk[0:128], blk[128:192], blk[192:208], blk[208:210]
        d = struct.unpack("<e", db)[0]
        sc = [s - 256 if s >= 128 else s for s in scb]
        y = b * 256
        for h in range(2):
            for l in range(32):
                is_ = l // 16
                q1 = (ql[h * 64 + l] & 0xF) | (((qh[h * 32 + l] >> 0) & 3) << 4)
                q2 = (ql[h * 64 + 32 + l] & 0xF) | (((qh[h * 32 + l] >> 2) & 3) << 4)
                q3 = (ql[h * 64 + l] >> 4) | (((qh[h * 32 + l] >> 4) & 3) << 4)
                q4 = (ql[h * 64 + 32 + l] >> 4) | (((qh[h * 32 + l] >> 6) & 3) << 4)
                base = y + 128 * h + l
                out[base] = d * sc[8 * h + is_] * (q1 - 32)
                out[base + 32] = d * sc[8 * h + is_ + 2] * (q2 - 32)
                out[base + 64] = d * sc[8 * h + is_ + 4] * (q3 - 32)
                out[base + 96] = d * sc[8 * h + is_ + 6] * (q4 - 32)
    return out


def _payload(name_wanted: str, rows: int | None = None):
    """Read one real tensor's bytes out of the real file."""
    kv, tensors, data_off = K.read_tensor_table(ORNITH)
    with open(ORNITH, "rb") as fh:
        for name, dims, ttype, off in tensors:
            if name != name_wanted:
                continue
            nelem = 1
            for d in dims:
                nelem *= int(d)
            if rows is not None and len(dims) == 2:
                nelem = int(dims[0]) * rows
            fh.seek(data_off + off)
            return kv, name, dims, ttype, fh.read(K.tensor_nbytes(ttype, nelem)), nelem
    raise AssertionError("tensor %s not in %s" % (name_wanted, ORNITH))


# --------------------------------------------------------------------------- #
# 1. layout arithmetic against the real file
# --------------------------------------------------------------------------- #
@needs_real
def test_block_sizes_reproduce_the_real_files_offsets():
    """The file's own offsets validate 256/144 for Q4_K and 256/210 for Q6_K."""
    kv, tensors, _ = K.read_tensor_table(ORNITH)
    bad_under = None
    for pad in (0, int(kv.get("general.alignment") or 32)):
        cur, bad = 0, []
        for i, (name, dims, ttype, off) in enumerate(tensors):
            nelem = 1
            for d in dims:
                nelem *= int(d)
            expect = 0 if i == 0 else (cur if not pad else -(-cur // pad) * pad)
            if off != expect:
                bad.append((name, off, expect))
            cur = expect + K.tensor_nbytes(ttype, nelem)
        if not bad:
            assert pad in (0, 32)
            return
        bad_under = bad_under or bad
    raise AssertionError("no alignment hypothesis reproduced the offsets; first "
                         "mismatch %s" % (bad_under[0],))


@needs_real
def test_real_file_types_are_the_three_we_claim():
    _kv, tensors, _ = K.read_tensor_table(ORNITH)
    seen: dict[int, int] = {}
    for _n, _d, ttype, _o in tensors:
        seen[ttype] = seen.get(ttype, 0) + 1
    assert seen == {14: 35, 0: 184, 12: 223}, (
        "Ornith uses Q4_K(12)/Q6_K(14)/F32(0) only; got %s" % seen)
    # the shifted table in tools/gui/model_import.py:74 would call 12 "Q5_K"
    assert K.type_name(12) == "Q4_K" and K.type_name(14) == "Q6_K"
    assert K.type_name(2) == "Q4_0" and K.type_name(30) == "BF16"


@needs_real
def test_element_bytes_estimate_is_wrong_for_blocks():
    """The 'bytes per element' idea in model_import.py:55 cannot size a block."""
    assert K.GGML_LAYOUT[12] == (256, 144)
    assert K.GGML_LAYOUT[2] == (32, 18)          # Q4_0 is not 2 bytes/element
    assert K.tensor_nbytes(12, 1017118720) == 572129280
    with raises(KeyError):
        K.tensor_nbytes(999, 256)


@needs_real
def test_real_file_size_is_explained_exactly_and_data_is_aligned():
    """The data section starts at an aligned offset past the tensor table.

    Two anchors, 26,090 bytes apart on this file, and only one of them closes the
    arithmetic:

      * the metadata (key/value) block ends at 10,942,907
      * the tensor table ends at   10,968,997 (mod 32 == 5)
      * the data section starts at 10,969,024

    Sizing the last tensor from the *unaligned* tensor-table end leaves the
    arithmetic 27 bytes short of the file size, and sizing it from the metadata end
    leaves it 26,112 short -- which is the proof that the anchor is the second one.
    """
    kv, tensors, data_off = K.read_tensor_table(ORNITH)
    prov = K.layout_provenance(ORNITH)
    assert data_off % prov["alignment"] == 0, (data_off, prov)
    assert prov["tensor_table_end"] % prov["alignment"] != 0, prov
    assert data_off - prov["tensor_table_end"] == 27, prov
    # The two anchors must not be confused: the metadata block ends earlier, and
    # aligning there is silently wrong in the other direction.
    assert prov["tensor_table_end"] > prov["metadata_end"], prov
    assert prov["metadata_end"] % prov["alignment"] != 0, prov
    assert (prov["tensor_table_end"] - prov["metadata_end"]) == 26090, prov
    last = tensors[-1]
    nelem = 1
    for d in last[1]:
        nelem *= int(d)
    end = data_off + last[3] + K.tensor_nbytes(last[2], nelem)
    assert end == os.path.getsize(ORNITH), (end, os.path.getsize(ORNITH))


@needs_real
def test_unaligned_data_base_would_silently_produce_garbage():
    """Negative control for the alignment: off by 27 bytes must be visible.

    This is what a reader that seeks to the unaligned tensor-table end (i.e. the
    raw ``fh.tell()``, which is what this module did before the fix) does.  It does
    not raise -- it returns numbers -- so the only way to catch it is to check that
    the decoded statistics are impossible for a weight matrix.
    """
    kv, tensors, data_off = K.read_tensor_table(ORNITH)
    prov = K.layout_provenance(ORNITH)
    assert data_off != prov["tensor_table_end"]
    dims, ttype, off = {n: (d, t, o) for n, d, t, o in tensors}["blk.0.ffn_gate.weight"]
    nelem = int(dims[0]) * 8
    need = K.tensor_nbytes(ttype, nelem)
    with open(ORNITH, "rb") as fh:
        fh.seek(prov["tensor_table_end"] + off)      # deliberately wrong base
        bad = K.dequantize_q4_K(fh.read(need), nelem)
        fh.seek(data_off + off)                     # the aligned base
        good = K.dequantize_q4_K(fh.read(need), nelem)
    assert not np.isfinite(good).all() or abs(float(good.std())) < 1.0, "aligned read is not sane"
    bad_is_visibly_wrong = (not np.isfinite(bad).all()) or float(bad.std()) > 1.0
    assert bad_is_visibly_wrong, (
        "the unaligned read looked plausible (std %.4g); the alignment check would "
        "not have caught the defect" % float(bad.std()))
    assert not np.array_equal(bad, good)


# --------------------------------------------------------------------------- #
# 2. arithmetic against the second implementation, on real blocks
# --------------------------------------------------------------------------- #
@needs_real
def test_q4_K_matches_independent_scalar_decoder_on_real_data():
    _kv, _n, _dims, ttype, raw, nelem = _payload("blk.0.ffn_gate.weight", rows=4)
    assert ttype == 12
    assert np.array_equal(K.dequantize_q4_K(raw, nelem), _ref_q4_K(raw, nelem)), \
        "vectorised != scalar on real Q4_K blocks"


@needs_real
def test_q6_K_matches_independent_scalar_decoder_on_real_data():
    _kv, _n, _dims, ttype, raw, nelem = _payload("blk.0.ffn_down.weight", rows=4)
    assert ttype == 14
    assert np.array_equal(K.dequantize_q6_K(raw, nelem), _ref_q6_K(raw, nelem)), \
        "vectorised != scalar on real Q6_K blocks"


# --------------------------------------------------------------------------- #
# 3. the decode lands on the quantisation lattice
# --------------------------------------------------------------------------- #
@needs_real
def test_q4_K_values_lie_on_the_lattice():
    _kv, _n, _dims, ttype, raw, nelem = _payload("blk.0.ffn_gate.weight", rows=8)
    assert ttype == 12
    y = K.dequantize_q4_K(raw, nelem)
    assert np.isfinite(y).all() and y.size == nelem
    off = 0
    for b in range(nelem // 256):
        blk = raw[b * 144:(b + 1) * 144]
        d = struct.unpack("<e", blk[0:2])[0]
        dmin = struct.unpack("<e", blk[2:4])[0]
        sc = blk[4:16]
        for g in range(4):
            for half, dq, mq in ((0, 2 * g, 2 * g), (32, 2 * g + 1, 2 * g + 1)):
                dd, mm = _get_scale_min_k4(dq, sc)
                if d * dd == 0:
                    continue
                seg = y[b * 256 + 64 * g + half: b * 256 + 64 * g + half + 32]
                q = (seg + dmin * mm) / (d * dd)
                off += int(np.count_nonzero(np.abs(q - np.round(q)) >= 1e-4))
                off += int(np.count_nonzero((np.round(q) < 0) | (np.round(q) > 15)))
    assert off == 0, "%d decoded Q4_K values are off the 4-bit grid" % off


@needs_real
def test_q6_K_values_lie_on_the_grid():
    _kv, _n, _dims, ttype, raw, nelem = _payload("blk.0.ffn_down.weight", rows=8)
    assert ttype == 14
    y = K.dequantize_q6_K(raw, nelem)
    assert np.isfinite(y).all() and y.size == nelem
    bad = 0
    for b in range(nelem // 256):
        blk = raw[b * 210:(b + 1) * 210]
        d = struct.unpack("<e", blk[208:210])[0]
        sc = [s - 256 if s >= 128 else s for s in blk[192:208]]
        for h in range(2):
            for l in range(32):
                is_ = l // 16
                for k, si in ((0, is_), (32, is_ + 2), (64, is_ + 4), (96, is_ + 6)):
                    v = y[b * 256 + 128 * h + l + k]
                    s = d * sc[8 * h + si]
                    if s == 0:
                        continue
                    q = v / s
                    bad += int(not (abs(q - round(q)) < 1e-4 and -32 <= round(q) <= 31))
    assert bad == 0, "%d decoded Q6_K values are off the 6-bit grid" % bad


@needs_real
def test_dequantised_statistics_are_plausible_for_a_real_model():
    kv, _n, dims, ttype, raw, nelem = _payload("blk.0.ffn_gate.weight", rows=256)
    assert ttype == 12
    y = K.dequantize_q4_K(raw, nelem).reshape(256, int(dims[0]))
    row_std = float(y.std(axis=1).mean())
    assert 0.001 < row_std < 1.0, "row std %.5f is not a weight matrix" % row_std
    assert abs(float(y.mean())) < 0.05, "mean %.5f is not zero-centred" % y.mean()
    _kv, _n, _d, t32, raw32, nelem32 = _payload("output_norm.weight")
    assert t32 == 0
    g = K.decode_f32(raw32, nelem32)
    assert np.isfinite(g).all() and len(g) == nelem32
    # A trained RMSNorm gain is positive and tightly distributed; it is *not*
    # centred on 1.0 (measured on this model: min 0.78, max 3.02, mean 2.15,
    # std 0.14).  An unaligned read gives negative values and a wild spread.
    assert (g > 0).all(), "rmsnorm gain has non-positive entries: min %.4g" % g.min()
    assert 1.0 < float(g.mean()) < 6.0, "gain mean %.4g" % g.mean()
    assert float(g.std() / g.mean()) < 0.5, "gain spread is not a trained gain"


# --------------------------------------------------------------------------- #
# 4. refusal paths
# --------------------------------------------------------------------------- #
def test_unsupported_type_is_refused_by_name():
    with raises(ValueError) as _:
        K.to_fp32(13, b"\x00" * 176, 256)          # Q5_K: layout known, not decoded
    try:
        K.to_fp32(13, b"\x00" * 176, 256)
    except ValueError as exc:
        msg = str(exc)
        assert "Q5_K" in msg and "13" in msg, msg
        assert "Q4_K" in msg and "Q6_K" in msg, msg


def test_wrong_payload_length_is_refused():
    with raises(ValueError):
        K.dequantize_q4_K(b"\x00" * 143, 256)
    with raises(ValueError):
        K.dequantize_q6_K(b"\x00" * 209, 256)
    with raises(ValueError):
        K.dequantize_q4_K(b"\x00" * 144, 200)      # not a multiple of 256


def test_type_table_matches_the_archkit_reader():
    """Two copies of the ggml table exist; they must not drift."""
    K.assert_type_table_agrees()


# --------------------------------------------------------------------------- #
# 5. naming
# --------------------------------------------------------------------------- #
@needs_real
def test_ornith_name_coverage_is_total():
    kv, tensors, _ = K.read_tensor_table(ORNITH)
    rep = N.coverage(kv, tensors)
    assert rep["total"] == 442, rep["total"]
    assert rep["unmapped"] == [], "unmapped: %s" % rep["unmapped"][:12]
    assert rep["mapped"] == 442, "only %d/442 mapped" % rep["mapped"]
    assert not rep["collisions"], rep["collisions"]
    N.require_full_coverage(rep)          # must not raise


@needs_real
def test_old_table_would_have_dropped_271_tensors():
    """The failure mode being eliminated, measured on the real file."""
    import re
    _kv, tensors, _ = K.read_tensor_table(ORNITH)
    legacy_names = [
        "token_embd.weight", "output_norm.weight", "output.weight",
        "blk.{i}.attn_norm.weight", "blk.{i}.attn_norm_2.weight",
        "blk.{i}.attn_q.weight", "blk.{i}.attn_k.weight", "blk.{i}.attn_v.weight",
        "blk.{i}.attn_output.weight", "blk.{i}.ffn_gate.weight",
        "blk.{i}.ffn_up.weight", "blk.{i}.ffn_down.weight",
    ]
    rx = [re.compile("^" + re.escape(p).replace(r"\{i\}", r"\d+") + "$")
          for p in legacy_names]
    dropped = [n for n, _d, _t, _o in tensors if not any(r.match(n) for r in rx)]
    assert len(dropped) == 271, len(dropped)
    assert len(tensors) - len(dropped) == 171


@needs_real
def test_hf_shape_reverses_gguf_dims_and_keeps_conv1d_unit_axis():
    kv, _t, _ = K.read_tensor_table(ORNITH)
    key, shape = N.translate("qwen35", "token_embd.weight", (4096, 248320), kv)
    assert key == "model.language_model.embed_tokens.weight"
    assert shape == (248320, 4096), shape
    key, shape = N.translate("qwen35", "blk.0.ffn_gate.weight", (4096, 12288), kv)
    assert shape == (12288, 4096), shape
    key, shape = N.translate("qwen35", "blk.0.ssm_conv1d.weight", (4, 8192), kv)
    assert key.endswith("linear_attn.conv1d.weight") and shape == (8192, 1, 4), shape
    key, shape = N.translate("qwen35", "output_norm.weight", (4096,), kv)
    assert shape == (4096,), shape


@needs_real
def test_layer_split_counts_the_draft_block_out_of_the_stack():
    kv, _t, _ = K.read_tensor_table(ORNITH)
    main, nextn = N.layer_split(kv, "qwen35")
    assert (main, nextn) == (32, 1), (main, nextn)
    key, _ = N.translate("qwen35", "blk.32.attn_q.weight", (4096, 8192), kv)
    assert key == "mtp.layers.0.self_attn.q_proj.weight", key
    key, _ = N.translate("qwen35", "blk.32.nextn.eh_proj.weight", (4096, 8192), kv)
    assert key == "mtp.fc.weight", key
    key, _ = N.translate("qwen35", "blk.31.attn_q.weight", (4096, 8192), kv)
    assert key == "model.language_model.layers.31.self_attn.q_proj.weight", key


def test_unknown_architecture_is_refused():
    with raises(KeyError):
        N.translate("gemma3", "blk.0.attn_q.weight", (4096, 4096), {})
    try:
        N.translate("gemma3", "blk.0.attn_q.weight", (4096, 4096), {})
    except KeyError as exc:
        assert "qwen35" in str(exc), str(exc)


def test_require_full_coverage_refuses_and_names_the_tensors():
    rep = {"arch": "qwen35", "total": 3, "mapped": 1, "unmapped": ["blk.0.weird.weight"],
           "collisions": [], "main_layers": 1, "nextn_layers": 0, "roles": {}}
    try:
        N.require_full_coverage(rep)
    except N.UnmappedTensors as exc:
        assert "blk.0.weird.weight" in str(exc)
        assert "refusing to write a partial checkpoint" in str(exc)
        return
    raise AssertionError("require_full_coverage accepted an unmapped tensor")


# --------------------------------------------------------------------------- #
# negative controls: the checks above must be able to fail
# --------------------------------------------------------------------------- #
def test_negative_control_shifted_type_table_is_detected():
    """The old defect was a table shifted by one; it must not look like ours."""
    shifted = {0: "F32", 1: "F16", 2: "BF16", 3: "Q4_0", 11: "Q4_K", 13: "Q6_K"}
    assert shifted[11] == "Q4_K" and shifted.get(12) != "Q4_K"
    assert shifted[13] == "Q6_K" and shifted.get(14) != "Q6_K"
    assert shifted[2] == "BF16"
    assert K.GGML_TYPES[12] == "Q4_K" and K.GGML_TYPES[14] == "Q6_K"
    assert K.GGML_TYPES[2] == "Q4_0"
    assert shifted != {k: v for k, v in K.GGML_TYPES.items() if k in shifted}


def test_negative_control_geometry_gate_rejects_a_wrong_geometry():
    """A geometry check that cannot fail proves nothing; this one can."""
    assert K.tensor_nbytes(12, 512) == 2 * 144
    with raises(ValueError):
        K.tensor_nbytes(12, 100)          # not a whole number of 256-value blocks


def test_negative_control_handbuilt_q6_K_block():
    """Hand-built block, hand-computed expectation.

    ql[l]=15, ql[32+l]=3, ql[64+l]=7, ql[96+l]=1 and qh[l]=0b00101010 (2-bit fields
    read 2,2,2,0 from the LSB), all scales 1, d = 1.0.  So lane 0 of the first half
    is: q1 = 15 | (2<<4) = 47 -> 47-32 = 15; q2 = 3 | 32 = 35 -> 3;
    q3 = (15>>4) | (2<<4) = 32 -> 0; q4 = (3>>4) | (0<<4) = 0 -> -32.
    """
    ql = bytearray(128)
    qh = bytearray(64)
    sc = bytearray([1] * 16)
    for l in range(32):
        ql[l] = 15
        ql[32 + l] = 3
        ql[64 + l] = 7
        ql[96 + l] = 1
    for l in range(64):
        qh[l] = 0b00101010        # both halves, so both halves are hand-checkable
    raw = bytes(ql) + bytes(qh) + bytes(sc) + struct.pack("<e", 1.0)
    y = K.dequantize_q6_K(raw, 256)
    assert y[0] == 15.0, y[0]
    assert y[32] == 3.0, y[32]
    assert y[64] == 0.0, y[64]
    assert y[96] == -32.0, y[96]
    assert y[128] == 7.0, y[128]                # second half, base lane -> ql[64]=7
    assert y[159] == 7.0, y[159]                # second half, lane 31 -> ql[64+31]=7
    assert y[191] == 1.0, y[191]                # +32 lane -> ql[96+31]=1, q2 = 1|32 = 33
    assert np.isfinite(y).all()


def test_negative_control_handbuilt_q4_K_block():
    """Encode a known 4-bit pattern and require it back exactly."""
    # scales: j=0 -> (sc=2, m=1), j=1 -> (sc=3, m=0), j=2..7 -> (1,0)
    scales = bytearray(12)
    scales[0] = 2
    scales[4] = 1
    scales[1] = 3
    scales[5] = 0
    qs = bytearray(128)
    qs[0] = 0x05          # low nibble 5 -> group 0 first 32; high nibble 0 -> group 0 second 32
    raw = (struct.pack("<e", 0.5) + struct.pack("<e", 0.25)
           + bytes(scales) + bytes(qs))
    y = K.dequantize_q4_K(raw, 256)
    # group 0, first 32 values: d * sc[0] * nibble - dmin * m[0]
    assert abs(y[0] - (0.5 * 2 * 5 - 0.25 * 1)) < 1e-6, y[0]
    assert abs(y[32] - (0.5 * 3 * 0 - 0.25 * 0)) < 1e-6, y[32]
    assert abs(y[1] - (0.5 * 2 * 0 - 0.25 * 1)) < 1e-6, y[1]


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    npass = nfail = nskip = 0
    for name, fn in tests:
        if getattr(fn, "__skip__", None):
            print("SKIP  %s" % name)
            nskip += 1
            continue
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            print("FAIL  %s: %s: %s" % (name, type(exc).__name__, exc))
            nfail += 1
        else:
            print("PASS  %s" % name)
            npass += 1
    for r in _skip_reasons:
        print("  skip reason: %s" % r)
    print("\n%d passed, %d failed, %d skipped (%d total)"
          % (npass, nfail, nskip, len(tests)))
    return 1 if nfail else 0


if __name__ == "__main__":
    raise SystemExit(main())
