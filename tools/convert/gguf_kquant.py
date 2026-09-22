#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gguf_kquant.py — ggml K-quant block formats: sizes and dequantisation.

Why this exists
---------------
``tools/convert/gguf_extract.py`` used to accept only ``F32``/``F16``/``BF16`` and
told the user to "download an F16/BF16 version".  That advice is not available for
every model: ``Ornith-1.5-9B-Q4_K_M.gguf`` is published as ``Q4_K``/``Q6_K`` only,
so without a K-quant reader there is no version of that model this tree can read.

What is here
------------
Only the ggml types actually present in that real file are *dequantised*:

===========  ====  ==================  ==================================
ggml id      name  block               measured count in Ornith Q4_K_M
===========  ====  ==================  ==================================
0            F32   -                   184 tensors
12           Q4_K  256 values / 144 B  223 tensors
14           Q6_K  256 values / 210 B   35 tensors
===========  ====  ==================  ==================================

The other K-quants get a *size* entry only, so that byte arithmetic on a file can
still be validated, and so that a file using them is refused by name instead of by
"unknown type id".  Adding one is a dequantiser plus a line in ``DEQUANTIZERS``.

The block layouts and the dequantisation arithmetic follow ggml's
``block_q4_K``/``block_q6_K`` and ``dequantize_row_q4_K``/``dequantize_row_q6_K``
(``ggml-quants.c``).  The 256-value block with a 144-byte Q4_K and 210-byte Q6_K
footprint is *independently* confirmed on the real file by
:func:`validate_layout_arithmetic`, which reproduces the file's own tensor offsets
from these block sizes alone -- that check does not use the dequantisers at all.

The ggml type-id table here is the same one already used correctly by
``tools/archkit/gguf_tensors.py:7-9`` (``GGML_TYPES``); :func:`assert_type_table_agrees`
cross-checks the two so they cannot drift apart.  ``tools/gui/model_import.py`` and
``tools/convert/gguf_extract.py`` used a table shifted by one (Q4_0 read as BF16,
Q4_K reported as Q5_K); this module is the correct one.
"""
from __future__ import annotations

import errno
import os
import struct
import sys
import time
from typing import Callable, Optional

import numpy as np

#: One-entry cache for the parsed tensor table of the most recent file.  The
#: table is a few MB and every caller wants it; re-parsing is wasted work, and
#: re-parsing with the values materialised is wasted *memory*.
_CACHE: dict = {}

#: GGUF's default tensor-data alignment when `general.alignment` is absent.
GGUF_DEFAULT_ALIGNMENT = 32

#: ggml type id -> canonical name.  Identical to tools/archkit/gguf_tensors.py.
GGML_TYPES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0",
    # Prism-private ids, read off the fork's own enum (PrismML-Eng/llama.cpp,
    # branch prism-v7, ggml/include/ggml.h:431 and :436):
    #   GGML_TYPE_Q1_0   = 41   -- 1-bit, one fp16 mean-|x| scale per 128 values
    #   GGML_TYPE_PTQ1_0 = 143  -- ternary, base-3 trits, one fp16 scale per 128
    #   GGML_TYPE_PQ2_0  = 142  -- Prism-private Q2_0 at group 128 (fork
    #                              ggml-common.h:202).  Added: the block
    #                              layout is read from the fork's own struct
    #                              and checked against the PTQ1_0 packing of
    #                              the same model, so it is measured, not guessed.
    # Upstream's 42 (Q2_0) is STILL not added: it carries two different block
    # layouts across releases, and a table that guesses one of them reads the
    # other silently.
    41: "Q1_0", 142: "PQ2_0", 143: "PTQ1_0",
}

#: ggml type id -> (values per block, bytes per block).  Non-block types use
#: (1, bytes per element).  F32/F16/BF16 are the types gguf_extract.py already
#: copied through unchanged.
GGML_LAYOUT: dict[int, tuple[int, int]] = {
    0: (1, 4), 1: (1, 2), 30: (1, 2),
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 292),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    # fork's own block sizes (ggml/src/ggml-common.h:185-190 and :214-220):
    #   block_q1_0   = { ggml_half d; uint8_t qs[16]; }              18 B / 128
    #   block_ptq1_0 = { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } 28 B / 128
    #   block_pq2_0  = { ggml_half d; uint8_t qs[32]; }              34 B / 128
    41: (128, 18), 142: (128, 34), 143: (128, 28),
}

QK_K = 256

#: Types this module can turn back into fp32.
DEQUANTIZERS: dict[int, "Callable[[bytes, int], np.ndarray]"] = {}


def type_name(type_id: int) -> str:
    return GGML_TYPES.get(type_id, "type%d" % type_id)


def tensor_nbytes(type_id: int, n_elements: int) -> int:
    """Exact stored size in bytes of ``n_elements`` values of this ggml type.

    This replaces the "bytes per element" estimate in
    ``tools/gui/model_import.py:55`` (``GGUF_ELEMENT_BYTES``), which is wrong for
    every block-quantised type: id 2 is Q4_0 (18 B / 32 values, not 2 B/elem) and
    the K-quants are 0.5-0.9 B/elem depending on the type, not a flat 1 or 2.
    """
    if type_id not in GGML_LAYOUT:
        raise KeyError("unknown ggml type id %d (%s)" % (type_id, type_name(type_id)))
    per_block, block_bytes = GGML_LAYOUT[type_id]
    if n_elements % per_block:
        raise ValueError(
            "ggml type %s stores %d values per block; %d is not a multiple of it"
            % (type_name(type_id), per_block, n_elements))
    return (n_elements // per_block) * block_bytes


# --------------------------------------------------------------------------- #
# Q4_K
# --------------------------------------------------------------------------- #
#: block_q4_K: fp16 d, fp16 dmin, 12 packed 6-bit (scale, min) pairs, 128 nibble B.
Q4K_BLOCK_BYTES = 144


def _q4k_scale_min(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``get_scale_min_k4`` for j = 0..7, vectorised over blocks.

    ``q`` is (n_blocks, 12) uint8.  Returns (scales, mins) as (n_blocks, 8) int32.
    For j < 4 the pair is a plain 6-bit field; for j >= 4 the high two bits come
    from the *other* half of the table, which is why both indices are needed.
    """
    n = q.shape[0]
    sc = np.empty((n, 8), np.int32)
    mn = np.empty((n, 8), np.int32)
    q = q.astype(np.int32)
    for j in range(4):
        sc[:, j] = q[:, j] & 63
        mn[:, j] = q[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (q[:, j + 4] & 0x0F) | ((q[:, j - 4] >> 6) << 4)
        mn[:, j] = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return sc, mn


def dequantize_q4_K(raw: bytes, n_elements: int) -> np.ndarray:
    """Q4_K -> fp32, in ggml element order.

    llamba.cpp's ``dequantize_row_q4_K`` walks the 256 values as four groups of 64;
    inside a group the low nibbles of 32 bytes come first, then the high nibbles,
    each with its own 6-bit scale and 6-bit minimum.
    """
    if n_elements % QK_K:
        raise ValueError("Q4_K needs a multiple of %d values, got %d" % (QK_K, n_elements))
    n_blocks = n_elements // QK_K
    want = n_blocks * Q4K_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q4_K payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q4K_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()
    dmin = blk[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    sc, mn = _q4k_scale_min(blk[:, 4:16])
    qs = blk[:, 16:144].reshape(n_blocks, 4, 32)

    # Intermediates are float64 and the single cast happens at the store.  fp16
    # mantissas are 11 bits, the 6-bit scale/min add 6 and the 4-bit code adds 4, so
    # every product here is exact in float64 and the result rounds identically to a
    # scalar decoder -- which is what lets the test compare bit-for-bit instead of
    # with a tolerance.
    df = d.astype(np.float64)
    dminf = dmin.astype(np.float64)
    scf = sc.astype(np.float64)
    mnf = mn.astype(np.float64)
    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    lo = (qs & 0x0F).astype(np.float64)
    hi = (qs >> 4).astype(np.float64)
    for g in range(4):
        d1 = (df * scf[:, 2 * g])[:, None]
        m1 = (dminf * mnf[:, 2 * g])[:, None]
        d2 = (df * scf[:, 2 * g + 1])[:, None]
        m2 = (dminf * mnf[:, 2 * g + 1])[:, None]
        out[:, 64 * g:64 * g + 32] = d1 * lo[:, g, :] - m1
        out[:, 64 * g + 32:64 * g + 64] = d2 * hi[:, g, :] - m2
    return out.ravel()


DEQUANTIZERS[12] = dequantize_q4_K


# --------------------------------------------------------------------------- #
# Q6_K
# --------------------------------------------------------------------------- #
#: block_q6_K: 128 low-nibble B, 64 two-bit-high B, 16 int8 scales, fp16 d.
Q6K_BLOCK_BYTES = 210


def dequantize_q6_K(raw: bytes, n_elements: int) -> np.ndarray:
    """Q6_K -> fp32, in ggml element order.

    Each 256-value block is two 128-value halves.  In a half, ``ql`` holds the low
    4 bits (``ql[l]`` and ``ql[l+32]``) and ``qh[l]`` the high 2 bits, two 2-bit
    fields per byte; the signed 6-bit value is ``bits - 32``.  The scale index is
    ``l // 16`` (0 or 1) stepping by 2 for the four interleaved outputs.
    """
    if n_elements % QK_K:
        raise ValueError("Q6_K needs a multiple of %d values, got %d" % (QK_K, n_elements))
    n_blocks = n_elements // QK_K
    want = n_blocks * Q6K_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q6_K payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q6K_BLOCK_BYTES)
    d = blk[:, 208:210].copy().view(np.float16).astype(np.float32).ravel()
    ql = blk[:, 0:128].reshape(n_blocks, 2, 64).astype(np.int32)
    qh = blk[:, 128:192].reshape(n_blocks, 2, 32).astype(np.int32)
    # int8 reinterpretation without relying on view() contiguity rules
    sc = np.where(blk[:, 192:208] >= 128, blk[:, 192:208].astype(np.int32) - 256,
                  blk[:, 192:208].astype(np.int32)).reshape(n_blocks, 2, 8)

    out = np.empty((n_blocks, QK_K), dtype=np.float32)
    for h in range(2):
        lo_l = ql[:, h, 0:32]
        lo_h = ql[:, h, 32:64]
        qhb = qh[:, h, :]
        for l in range(32):
            isrc = l // 16
            q1 = (lo_l[:, l] & 0x0F) | (((qhb[:, l] >> 0) & 3) << 4)
            q2 = (lo_h[:, l] & 0x0F) | (((qhb[:, l] >> 2) & 3) << 4)
            q3 = (lo_l[:, l] >> 4) | (((qhb[:, l] >> 4) & 3) << 4)
            q4 = (lo_h[:, l] >> 4) | (((qhb[:, l] >> 6) & 3) << 4)
            base = 128 * h + l
            out[:, base] = d * sc[:, h, isrc + 0] * (q1 - 32)
            out[:, base + 32] = d * sc[:, h, isrc + 2] * (q2 - 32)
            out[:, base + 64] = d * sc[:, h, isrc + 4] * (q3 - 32)
            out[:, base + 96] = d * sc[:, h, isrc + 6] * (q4 - 32)
    return out.ravel()


DEQUANTIZERS[14] = dequantize_q6_K


# --------------------------------------------------------------------------- #
# pass-through types (already supported by gguf_extract.py before this module)
# --------------------------------------------------------------------------- #
def decode_f32(raw: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f4", count=n_elements).astype(np.float32)


def decode_f16(raw: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(raw, dtype="<f2", count=n_elements).astype(np.float32)


def decode_bf16(raw: bytes, n_elements: int) -> np.ndarray:
    u = np.frombuffer(raw, dtype="<u2", count=n_elements).astype(np.uint32) << 16
    return u.view(np.float32)


DEQUANTIZERS[0] = decode_f32
DEQUANTIZERS[1] = decode_f16
DEQUANTIZERS[30] = decode_bf16

# --------------------------------------------------------------------------- #
# Q1_0 / PTQ1_0 -- Prism-private block formats (fork type ids 41 and 143)
# --------------------------------------------------------------------------- #
# The two block layouts and both dequantisation loops are read from the fork,
# PrismML-Eng/llama.cpp branch prism-v7 (tip c1abda39458458eb):
#   ggml/src/ggml-common.h:185  #define QK1_0 128
#   ggml/src/ggml-common.h:186  block_q1_0   { ggml_half d; uint8_t qs[16]; }
#   ggml/src/ggml-quants.c:40   quantize_row_q1_0_ref   d = sum|x|/128, one sign bit each
#   ggml/src/ggml-quants.c:454  dequantize_row_q1_0     bit ? +d : -d
#   ggml/src/ggml-common.h:214  #define QK_PTQ1_0 128
#   ggml/src/ggml-common.h:215  block_ptq1_0 { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; }
#   ggml/src/ggml-quants.c:2255 dequantize_row_ptq1_0
# Both are affine with a fixed code alphabet: {+1,-1}*d for Q1_0 and
# {-1,0,+1}*d for PTQ1_0.  Measured against the published files in this record:
# the file's data section closes to the byte from these two block sizes alone, and
# every one of the 5,392,656 qs bytes sampled from the ternary file re-encodes
# through the fork's own ceil(q*256/243) map (see dl/type41/).
QK1_0 = 128
Q1_0_BLOCK_BYTES = 18
QK_PTQ1_0 = 128
PTQ1_0_BLOCK_BYTES = 28
QK_PQ2_0 = 128
PQ2_0_BLOCK_BYTES = 34

#: (values per byte group, byte offset inside qs) for the ternary stages.  For this
#: 24-byte qs the fork's 32/16/8 staging resolves to exactly one 16-byte stage and
#: one 8-byte stage, which is the whole payload: 16*5 + 8*5 = 120 values, plus 2*4
#: from qh = 128.  (The same staging on TQ1_0's 48-byte qs gives its 32-then-16.)
_PTQ1_0_STAGES = ((16, 0), (8, 16))


def decode_q1_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Q1_0 -> fp32.  Bit j of the 16-byte bitmap is byte j//8, bit j%8; set = +d."""
    if n_elements % QK1_0:
        raise ValueError("Q1_0 needs a multiple of %d values, got %d" % (QK1_0, n_elements))
    n_blocks = n_elements // QK1_0
    want = n_blocks * Q1_0_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q1_0 payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q1_0_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    bits = np.unpackbits(blk[:, 2:18], axis=1, bitorder="little").astype(np.float32)
    return np.where(bits > 0, d[:, None], -d[:, None]).astype(np.float32).ravel()


def decode_ptq1_0(raw: bytes, n_elements: int) -> np.ndarray:
    """PTQ1_0 -> fp32, exactly the fork's arithmetic.

    Per byte the fork recovers digit n as ``((b * 3**n) mod 256 * 3) >> 8`` -- the
    truncation to uint8_t before the shift is part of the stored format, not a
    detail: dropping it decodes every byte differently.  Output order is the
    decoder's: stage by stage, n outer and m inner, then the 4x2 qh group.
    """
    if n_elements % QK_PTQ1_0:
        raise ValueError("PTQ1_0 needs a multiple of %d values, got %d"
                         % (QK_PTQ1_0, n_elements))
    n_blocks = n_elements // QK_PTQ1_0
    want = n_blocks * PTQ1_0_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("PTQ1_0 payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, PTQ1_0_BLOCK_BYTES)
    d = blk[:, 26:28].copy().view("<f2").astype(np.float32).ravel()
    qs = blk[:, 0:24].astype(np.uint32)
    qh = blk[:, 24:26].astype(np.uint32)
    pow3 = np.array([1, 3, 9, 27, 81], dtype=np.uint32)
    parts = []
    for count, j0 in _PTQ1_0_STAGES:
        for n in range(5):
            q = (qs[:, j0:j0 + count] * pow3[n]) & 0xFF       # uint8_t truncation
            parts.append((((q * 3) >> 8).astype(np.int32) - 1))
    for n in range(4):
        for h in range(2):
            q = (qh[:, h:h + 1] * pow3[n]) & 0xFF
            parts.append((((q * 3) >> 8).astype(np.int32) - 1))
    trits = np.concatenate(parts, axis=1)
    if trits.shape[1] != n_elements // n_blocks:
        raise AssertionError("PTQ1_0 decoder produced %d of %d values"
                             % (trits.shape[1], n_elements // n_blocks))
    return (trits.astype(np.float32) * d[:, None]).astype(np.float32).ravel()


def decode_pq2_0(raw: bytes, n_elements: int) -> np.ndarray:
    """PQ2_0 -> fp32, exactly the fork's arithmetic.

    The fork's loop (ggml/src/ggml-quants.c, dequantize_row_pq2_0)::

        q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
        y[i*qk + j] = ((int) q - 1) * d;      // 00=-1, 01=0, 10=+1, 11=+2

    and its block (ggml/src/ggml-common.h:202-207)::

        #define QK_PQ2_0 128
        typedef struct { ggml_half d; uint8_t qs[QK_PQ2_0 / 4]; } block_pq2_0;

    The scale is FIRST, unlike PTQ1_0's trailing one, and the two bits run
    low-first inside each byte.  The codec carries four levels; a ternary
    checkpoint uses the first three and never emits 11.

    Checked against the PTQ1_0 packing of the same model -- same weights,
    different packing -- value for value over 5,242,880 values with zero
    mismatches, while `d` last, msb-first and `q` without the -1 each move
    3.5 M values or more (dl/_orch/pq2match2.py).
    """
    if n_elements % QK_PQ2_0:
        raise ValueError("PQ2_0 needs a multiple of %d values, got %d"
                         % (QK_PQ2_0, n_elements))
    n_blocks = n_elements // QK_PQ2_0
    want = n_blocks * PQ2_0_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("PQ2_0 payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, PQ2_0_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    qs = blk[:, 2:34].astype(np.uint32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint32)
    codes = np.stack([(qs >> s) & 0x03 for s in shifts], axis=2).reshape(n_blocks, -1)
    if codes.shape[1] != n_elements // n_blocks:
        raise AssertionError("PQ2_0 decoder produced %d of %d values"
                             % (codes.shape[1], n_elements // n_blocks))
    return ((codes.astype(np.int32) - 1).astype(np.float32)
            * d[:, None]).astype(np.float32).ravel()


DEQUANTIZERS[41] = decode_q1_0
DEQUANTIZERS[142] = decode_pq2_0
DEQUANTIZERS[143] = decode_ptq1_0


# --------------------------------------------------------------------------- #
# Q8_0 (upstream ggml id 8, which the Prism fork keeps)
# --------------------------------------------------------------------------- #
#   ggml/src/ggml-common.h   #define QK8_0 32
#                             typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;
#   ggml/src/ggml-quants.c   dequantize_row_q8_0:  y[i*QK8_0 + j] = x[i].qs[j]*d;
#
# The scale is FIRST and the 32 elements are signed bytes in qs order.  The layout
# is not a guess made here: it is the one this module already stores
# (``GGML_LAYOUT[8] == (32, 34)``) and the one ``validate_layout_arithmetic``
# reproduces a real file's tensor offsets from, so a decoder that disagreed with it
# could not also pass that check.
#
# This decoder is here for the draft block of the MTP-bearing Bonsai packs:
# ``Bonsai-2-27B-PQ2_0-MTP.gguf`` stores ``blk.64.attn_k.weight`` and
# ``blk.64.attn_v.weight`` as Q8_0 and nothing else in the file, and both
# ``tools/convert/gguf_fold_back.py`` and ``tools/convert/gguf_extract.py`` gate on
# membership in ``DEQUANTIZERS``, so without this entry neither can reach the file
# (measured 2026-09-20, dl/mtpengine/).
QK8_0 = 32
Q8_0_BLOCK_BYTES = 34


def decode_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Q8_0 -> fp32: per 32-value block, an fp16 delta then 32 signed bytes."""
    if n_elements % QK8_0:
        raise ValueError("Q8_0 needs a multiple of %d values, got %d" % (QK8_0, n_elements))
    n_blocks = n_elements // QK8_0
    want = n_blocks * Q8_0_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q8_0 payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    q = blk[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (q * d[:, None]).astype(np.float32).ravel()


DEQUANTIZERS[8] = decode_q8_0

#: Human-usable spelling of what this module can read, for refusal messages.
SUPPORTED_NAMES = "/".join(sorted(GGML_TYPES[t] for t in DEQUANTIZERS))
#: Types whose byte layout is known but which cannot be decoded yet.
KNOWN_UNSUPPORTED = tuple(sorted(
    t for t in GGML_LAYOUT if t not in DEQUANTIZERS
    and GGML_TYPES.get(t, "").endswith(("_K", "_0", "_1", "_NL", "_XS", "_S", "_M"))
))


def to_fp32(type_id: int, raw: bytes, n_elements: int) -> np.ndarray:
    """Dequantise one tensor payload to fp32.  Refuses by name, never silently."""
    fn = DEQUANTIZERS.get(type_id)
    if fn is None:
        raise ValueError(
            "cannot read ggml type %d (%s); this reader decodes %s. A type whose "
            "layout is known here but which is not decoded yet: %s"
            % (type_id, type_name(type_id), SUPPORTED_NAMES,
               ", ".join("%s(%d)" % (type_name(t), t) for t in KNOWN_UNSUPPORTED)
               or "none"))
    return fn(raw, n_elements)


def fp32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """fp32 -> bf16 bytes by high-half truncation (what gguf_extract.py emitted)."""
    return (np.ascontiguousarray(arr, dtype=np.float32).view("<u4") >> 16) \
        .astype("<u2").tobytes()


# --------------------------------------------------------------------------- #
# verification helpers (used by tests/convert/test_gguf_kquant.py)
# --------------------------------------------------------------------------- #
class GGUFArray:
    """A metadata array, summarised instead of materialised.

    ``tokenizer.ggml.tokens`` on a 248,320-token model is a list of 248,320 Python
    strings (~200 MB) and is of no interest to a tensor reader.  Only the element
    type, the count and the first few values are kept; the rest is skipped in the
    stream.  ``len()`` still answers the count, which is what ``vocab_size`` needs.
    """

    __slots__ = ("elem_type", "count", "first")

    def __init__(self, elem_type: int, count: int, first: list):
        self.elem_type = elem_type
        self.count = count
        self.first = first

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return "GGUFArray(type=%s, count=%d, first=%r)" % (
            GGML_TYPES.get(self.elem_type, self.elem_type), self.count, self.first[:4])


#: Element types kept verbatim (they are small); larger arrays are only summarised.
_VERBATIM_ARRAY = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12)

#: Fixed width of the scalar metadata types, so an array of them can be skipped
#: with one seek instead of one call per element.
_SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


#: Kernel errors a read can return TRANSIENTLY on this machine's /mnt/c relay.
#: Measured 2026-09-14 (two separate runs of the self-test): ``f.read(8)`` inside the
#: metadata walk raised ``OSError: [Errno 12] Cannot allocate memory`` at
#: ``gguf_tensors.py:18`` and at ``gguf_kquant.py:355``.  The failing call is EIGHT
#: BYTES, and an 8-byte read on a BufferedReader is a fill of its existing 8 KiB
#: buffer -- so neither the requested size nor any ulimit is the variable.  The same
#: file parsed in another process in the same minute, and a byte-identical 12 MB head
#: on ext4 never failed: the failure is a transient property of the mount under guest
#: memory pressure.  A retry is therefore the right shape, not a smaller/larger read.
TRANSIENT_READ_ERRNOS = frozenset((errno.EINTR, errno.EAGAIN, errno.ENOMEM, errno.EBUSY))
READ_ATTEMPTS = 5


def transient_read(fh, n):
    """``fh.read(n)``, retrying a transient kernel error a bounded number of times.

    Not silent: the first retry says so on stderr, and when the attempts run out the
    original error propagates unchanged -- an absorbed error becomes a *logged* one,
    and an unabsorbable one never turns into a short read (a short read would
    desynchronise the metadata walk and yield garbage names with no exception at all).
    """
    for attempt in range(READ_ATTEMPTS):
        try:
            return fh.read(n)
        except OSError as exc:
            if exc.errno not in TRANSIENT_READ_ERRNOS or attempt + 1 >= READ_ATTEMPTS:
                raise
            if attempt == 0:
                sys.stderr.write(
                    "[gguf] transient %s on a %s B read; retrying up to %d times\n"
                    % (exc, n, READ_ATTEMPTS - 1))
            time.sleep(0.25 * (attempt + 1))


class _RetryingReader:
    """Read-only wrapper making :func:`transient_read` the only way to read.

    ``scan``/``_parse`` open one file and then do hundreds of thousands of tiny reads,
    so wrapping the handle is the smallest change that covers all of them -- including
    the ones added later.
    """

    def __init__(self, fh):
        self._fh = fh

    def read(self, n=-1):
        return transient_read(self._fh, n)

    def seek(self, *args):
        return self._fh.seek(*args)

    def tell(self):
        return self._fh.tell()

    def close(self):
        return self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._fh.close()


def _parse(path: str):
    """Parse one GGUF header, metadata block and tensor table.  No caching here.

    Returns ``(kv, tensors, data_offset, provenance)``; ``provenance`` records where
    the data section starts and why (see the alignment note near the end).

    Header is magic(4) + version(u32) + tensor_count(u64) + kv_count(u64) = 24 B;
    reading it as ``"<IQI"`` (20 B) is the defect in
    ``tools/gui/model_import.py:318`` and the pre-fix
    ``tools/convert/gguf_extract.py:53``.
    """
    with open(path, "rb") as fh:
        # Every read below goes through the retrying wrapper: this parser walks
        # ~250k metadata strings with one 8-byte read each, which is where the
        # measured transient ENOMEM landed (gguf_kquant.py:355, 15:39).
        fh = _RetryingReader(fh)
        if fh.read(4) != b"GGUF":
            raise ValueError("%s is not a GGUF file" % path)
        head = fh.read(20)
        if len(head) != 20:
            raise ValueError("GGUF header truncated (%d of 24 bytes)" % (4 + len(head)))
        _version, n_tensors, n_kv = struct.unpack("<IQQ", head)

        def rd_str():
            n = struct.unpack("<Q", fh.read(8))[0]
            if n > (1 << 24):
                raise ValueError("metadata string length %d is implausible" % n)
            return fh.read(n).decode("utf-8", "replace")

        def skip(t, depth=0):
            """Advance past one metadata value without materialising it.

            Strings seek instead of read+decode: ``tokenizer.ggml.tokens`` is a
            248,320-entry string array on a real model and decoding it just to
            discard it is both slow and -- measured on this machine -- enough to
            fail with ENOMEM under memory pressure.
            """
            if depth > 4:
                raise ValueError("metadata nesting too deep")
            if t in (0, 1, 7):
                fh.seek(1, 1)
            elif t in (2, 3):
                fh.seek(2, 1)
            elif t in (4, 5, 6):
                fh.seek(4, 1)
            elif t == 8:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n > (1 << 24):
                    raise ValueError("metadata string length %d is implausible" % n)
                fh.seek(n, 1)
            elif t == 9:
                at, cnt = struct.unpack("<IQ", fh.read(12))
                if cnt > (1 << 22):
                    raise ValueError("metadata array of %d elements is implausible" % cnt)
                if at in (0, 1, 7) or at in (2, 3) or at in (4, 5, 6) or at in (10, 11, 12):
                    fh.seek(cnt * (_SCALAR_BYTES[at]), 1)      # fixed-width: one seek
                else:
                    for _ in range(cnt):
                        skip(at, depth + 1)
            elif t in (10, 11, 12):
                fh.seek(8, 1)
            else:
                raise ValueError("unknown GGUF metadata type %d" % t)

        def read_val(t):
            if t in (0, 1, 7):
                return fh.read(1)[0]
            if t in (2, 3):
                return struct.unpack("<h" if t == 3 else "<H", fh.read(2))[0]
            if t in (4, 5, 6):
                return struct.unpack("<i" if t == 5 else ("<f" if t == 6 else "<I"),
                                     fh.read(4))[0]
            if t == 8:
                return rd_str()
            if t == 9:
                at, cnt = struct.unpack("<IQ", fh.read(12))
                if cnt > (1 << 22):
                    raise ValueError("metadata array of %d elements is implausible" % cnt)
                keep = cnt if (at in _VERBATIM_ARRAY and cnt <= 4096) else min(cnt, 8)
                first = [read_val(at) for _ in range(keep)]
                remaining = cnt - keep
                if remaining:
                    if at in _SCALAR_BYTES:
                        fh.seek(remaining * _SCALAR_BYTES[at], 1)
                    else:
                        for _ in range(remaining):
                            skip(at)
                return GGUFArray(at, cnt, first)
            if t in (10, 11, 12):
                return struct.unpack("<q" if t == 11 else ("<d" if t == 12 else "<Q"),
                                     fh.read(8))[0]
            raise ValueError("unknown GGUF metadata type %d" % t)

        kv = {}
        for _ in range(n_kv):
            key = rd_str()
            (t,) = struct.unpack("<I", fh.read(4))
            kv[key] = read_val(t)
        # Where the metadata block itself ends.  Recorded only so the two anchors can
        # be told apart: the alignment applies at the end of the *tensor table*, not
        # here (see the note below the tensor loop).
        metadata_end = fh.tell()
        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            n_dims = struct.unpack("<I", fh.read(4))[0]
            dims = struct.unpack("<" + "Q" * n_dims, fh.read(8 * n_dims))
            (ttype,) = struct.unpack("<I", fh.read(4))
            (off,) = struct.unpack("<Q", fh.read(8))
            tensors.append((name, dims, ttype, off))
        # The tensor-data section starts at the first `general.alignment`-aligned
        # offset at or after the end of the **tensor table** -- NOT at the raw
        # `fh.tell()`, and NOT after the metadata block.  Tensor `offset` values are
        # relative to that base.
        #
        # Measured on Ornith-1.5-9B-Q4_K_M.gguf and independently reproduced with
        # /home/user/scratch/impv/bin/ornith_census.py (a spec-strict 24-byte-header
        # reader that shares no code with this module):
        #   metadata block ends at      10,942,907
        #   tensor table (442 entries)  10,968,997  <- this is the anchor, % 32 == 5
        #   data section starts at      10,969,024  <- 27 bytes later
        # The two anchors are 26,090 bytes apart and only the second one closes the
        # arithmetic.  A reader that uses the raw `fh.tell()` is 27 bytes short, and
        # then *every* tensor it decodes is shifted by 27 bytes -- the 2-byte fp16
        # scale fields land on payload bytes, values come out as garbage (some even
        # decode to NaN), and nothing raises.  The giveaway is arithmetic: with the
        # unaligned base, `data_offset + last.offset + last.nbytes` falls exactly 27
        # bytes short of the file size, and for this file
        # 5,780,090,816 - 5,780,090,789 == 27.  Aligning at the *metadata* end
        # instead is equally wrong in the other direction: it would start the data
        # at 10,942,912, 26,112 bytes early, and just as silently.
        alignment = int(kv.get("general.alignment") or GGUF_DEFAULT_ALIGNMENT)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("general.alignment=%d is not a power of two" % alignment)
        tensor_table_end = fh.tell()
        data_offset = -(-tensor_table_end // alignment) * alignment
        return (kv, tensors, data_offset,
                {"metadata_end": metadata_end, "tensor_table_end": tensor_table_end,
                 "alignment": alignment, "data_offset": data_offset})


def _cached(path: str):
    """Parse a GGUF once per (path, size).

    The tensor table is a few MB and every caller wants it; re-parsing is wasted
    work, and re-parsing with the metadata values materialised is wasted memory.
    """
    key = (path, os.path.getsize(path))
    hit = _CACHE.get(key)
    if hit is None:
        _CACHE.clear()
        _CACHE[key] = hit = _parse(path)
    return hit


def read_tensor_table(path: str, cache: bool = True):
    """Spec-correct GGUF header + metadata skip + tensor table read.

    Returns ``(kv, tensors, data_offset)``: ``tensors`` is a list of
    ``(name, dims, type_id, offset)``, and ``data_offset`` is the *aligned* start of
    the tensor-data section that those offsets are relative to.
    """
    return (_cached(path) if cache else _parse(path))[:3]


def layout_provenance(path: str) -> dict:
    """``{'metadata_end', 'tensor_table_end', 'alignment', 'data_offset'}`` for one GGUF.

    Kept separate so a caller can state *why* the data offset is what it is: the
    27-byte difference between ``tensor_table_end`` and ``data_offset`` on a real
    file is the whole point.  ``metadata_end`` is the end of the key/value block,
    26,090 bytes earlier on that file; it is **not** the alignment anchor, and
    aligning there instead is silently wrong in the other direction.
    """
    return dict(_cached(path)[3])


def validate_layout_arithmetic(path: str, verbose: bool = False) -> dict:
    """Reproduce the file's own tensor offsets from the block sizes alone.

    This is the independent check that the 256-value/144-byte Q4_K and
    210-byte Q6_K layouts are the ones this file was written with: it uses only
    the offset column of the tensor table and :func:`tensor_nbytes`, never the
    dequantisers.  Offsets are written in tensor-table order with
    ``general.alignment`` padding, so a wrong block size shows up as a mismatch.
    """
    kv, tensors, data_offset = read_tensor_table(path)
    align = int(kv.get("general.alignment") or 32)

    def attempt(pad_to: int):
        cur, bad = 0, []
        rows = []
        for i, (name, dims, ttype, off) in enumerate(tensors):
            nelem = 1
            for d in dims:
                nelem *= int(d)
            expect = 0 if i == 0 else (cur if not pad_to else -(-cur // pad_to) * pad_to)
            try:
                need = tensor_nbytes(ttype, nelem)
            except (KeyError, ValueError) as exc:
                rows.append((name, ttype, nelem, None, off, "no layout: %s" % exc))
                cur = expect
                continue
            rows.append((name, ttype, nelem, need, off,
                         "ok" if off == expect else "expected %d got %d" % (expect, off)))
            if off != expect:
                bad.append(rows[-1])
            cur = expect + need
        return rows, bad

    # Both hypotheses are tried: a writer that pads every tensor start to
    # `general.alignment`, and one that packs them contiguously.  Ornith's file is
    # the contiguous kind (210-byte Q6_K blocks are not 32-byte multiples), so
    # reporting which one holds is the point.
    for pad_to in (0, align):
        rows, bad = attempt(pad_to)
        if not bad:
            if verbose:
                for r in rows:
                    print("  %-60s %-6s <nelem %d> need=%s off=%d %s"
                          % (r[0], type_name(r[1]), r[2], r[3], r[4], r[5]))
            return {"tensors": len(tensors), "data_offset": data_offset,
                    "checked": sum(1 for r in rows if r[3] is not None),
                    "padding": pad_to, "mismatches": [], "ok": True}
    return {"tensors": len(tensors), "data_offset": data_offset,
            "checked": sum(1 for r in rows if r[3] is not None),
            "padding": None, "mismatches": bad, "ok": False,
            "first_mismatch": bad[0] if bad else None}


def assert_type_table_agrees() -> None:
    """The type table here and tools/archkit/gguf_tensors.py's must be identical."""
    sys_path_added = False
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        archkit = root / "tools" / "archkit"
        if str(archkit) not in sys.path:
            sys.path.insert(0, str(archkit))
            sys_path_added = True
        import gguf_tensors  # type: ignore
    except Exception:
        return  # archkit not present; nothing to cross-check
    finally:
        if sys_path_added:
            pass
    theirs = dict(gguf_tensors.GGML_TYPES)
    mine = {k: v for k, v in GGML_TYPES.items() if k in theirs}
    if mine != theirs:
        raise AssertionError(
            "ggml type table drift: tools/archkit/gguf_tensors.py says %s, "
            "gguf_kquant.py says %s" % (theirs, mine))
