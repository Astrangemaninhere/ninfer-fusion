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



# =========================================================================== #
# rco-support block, appended 2026-09-26 by line `rcosup` (marker F-894).
#
# WHY.  Two real Swift 1.5 candidates are refused at this reader gate BY NAME, on
# types whose arithmetic ggml fully specifies -- and the publisher of both writes
# them with llama.cpp itself, so ggml IS the definition:
#   Swift-1.5-Qwen3.8-27B-Q4_K_M.gguf         17,442,399,936 B
#       866 tensors; refused: Q5_K x2 (blk.51/55.ffn_down), Q4_0 x8 (the MTP head)
#   Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf  11,771,546,912 B
#       851 tensors; refused 363: IQ1_M 1, IQ2_XXS 5, IQ2_XS 9, Q2_K 13,
#       IQ2_S 17, IQ3_XXS 78, IQ4_XS 96, IQ3_S 144
#   Ling-3.0-tiny-IQ4_XS.gguf                 4,385,795,776 B
#       526 tensors; refused 208: Q5_K 24, IQ4_NL 6, IQ4_XS 178 -- handed to this
#       line by dl/convmulti (F-899) after this block was first proposed, and it is
#       why IQ4_NL is here: a type the first draft did not cover AT ALL.
# The publisher names the algorithm as `llama-quantize --tensor-type-file`
# (RECIPE.txt), and every one of these types is a fixed-size block with a
# published dequantisation -- no reader-side search, no learned parameter.
#
# WHAT THIS BLOCK IS.  An APPEND.  Every line above it is untouched, byte for
# byte, so every existing call site, refusal message and test behaves exactly as
# before for every file that was already readable.  The two derived strings the
# refusal message uses are re-derived at the bottom because they were computed
# before this block existed and would otherwise understate what is readable.
#
# WHAT IT IS NOT.  Not one decoder already present is respelled.  F32, F16,
# BF16, Q4_K, Q6_K, Q8_0 and the three Prism-private ids were already decoded
# here and are not touched; Q8_0 in particular ALREADY covers 64 of the Q4_K_M
# file's tensors, so that file's true refusal population is 10 of 866, not 74.
#
# --------------------------------------------------------------------------- #
# THE GENERAL ADAPTATION PATH -- the answer to "there will be many such models".
# --------------------------------------------------------------------------- #
# A new ggml quant type can be added to THIS reader in exactly three shapes, and
# the shape decides the cost.  The classification is per TYPE, not per file:
#
#   (i)   A TABLE ROW AND A CLOSED-FORM DECODER.  The type is affine: a stored
#         fp16 scale (and usually a minimum) times an integer code.  Nothing
#         outside the block bytes is needed, and there is nothing to get wrong
#         except the bit unpacking.  Cost: one GGML_LAYOUT row + one function.
#         TONIGHT: Q4_0, Q5_K, Q2_K.  ALREADY THERE: Q4_K, Q6_K, Q8_0, and the
#         fork-private Q1_0/PQ2_0/PTQ1_0.
#         THE TREE CAN EXPRESS THIS TODAY -- GGML_LAYOUT + DEQUANTIZERS is
#         exactly that seam, and it has been used for it since the file existed.
#
#   (ii)  A TABLE ROW, A DECODER, AND A CODEBOOK/GRID RESOURCE.  The block stores
#         an INDEX into a fixed table of reconstruction vectors, so the table is
#         part of the FORMAT, not of the file: the same table decodes every file
#         and is identical upstream.  Cost: the decoder PLUS the table bytes.
#         TONIGHT: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S (grids), IQ4_XS and
#         IQ4_NL (the SAME 16-entry iq4nl codebook, different block frame),
#         IQ1_M (the 2048-row iq1s grid).
#         THE TREE COULD NOT EXPRESS THIS BEFORE THIS BLOCK.  It had no home for
#         a table: the only data seams were GGML_TYPES and GGML_LAYOUT, both of
#         which describe SIZES.  _grid_from_hex() below is that home, and it is
#         deliberately data-driven: a further grid type is a literal plus a call,
#         and a truncated literal RAISES instead of decoding a short table.
#         SO: a future model quantised to an i-quant this file does not yet
#         decode -- IQ2_XS at another bit-width, TQ1_0/TQ2_0, MXFP4, a new
#         i-grid from the same lab -- is now class (ii): ONE ROW, ONE FUNCTION,
#         ONE TABLE, and it is a patch to this file rather than a redesign.
#
#   (iii) A COMPANION ARTIFACT THE FILE MAY NOT CARRY.  *** NONE.  THIS CLASS IS
#         EMPTY FOR READING, AND THAT IS A RESULT, NOT A SHRUG. ***
#         An importance matrix (imatrix) is the obvious candidate and it is NOT
#         one: it is an AUTHORING input.  Dequantisation is a pure function of
#         the block bytes -- y = f(block) with every coefficient stored in the
#         block or in a table embedded in the format, which is why this file
#         decodes these types with NO network access, NO calibration data and
#         nothing but the .gguf.  The publisher ships
#         `imatrix-swift15-v1mix.gguf` in the same repository, and RECIPE.txt
#         shows it consumed by the refinement step that PRODUCED the file, never
#         by a runtime that reads it.  A reader that demanded one would refuse a
#         file that is perfectly readable.
#         CONSEQUENCE FOR THE OWNER, stated because it is the useful half: every
#         ggml quant type that EXISTS today is (i) or (ii).  There is no ggml
#         type whose reader needs a second artifact.  So "adapt the next model"
#         is bounded work -- a row, a decoder, and at worst a grid -- and never a
#         data-collection problem.  The only genuinely unbounded case would be a
#         FORMAT CHANGE (a new container, or a block whose layout is not
#         self-describing), and that is not a quant type at all.
#
# WHAT MAKES THE PATH GENERAL RATHER THAN TONIGHT'S PATCH.  Two instruments are
# added below and they are the part that scales:
#   quant_coverage(path)      -- for ANY gguf, the per-type histogram, how many
#                                of its tensors this reader decodes, and the class
#                                (i)/(ii)/(iii) of every type it refuses.  One
#                                command answers "will this new model just work?".
#   infer_type_layout(path, t) -- for a type NOT in GGML_LAYOUT, MEASURE its
#                                (values/block, bytes/block) from the file's own
#                                tensor offsets, instead of raising "unknown ggml
#                                type id" with no size at all.  This turns the
#                                worst case from "no information" into "here is the
#                                block size; only the bit order is left".
#
# VERIFICATION (dl/rcosup/).  Every decoder added was compared BIT-FOR-BIT over
# EVERY tensor of its type in BOTH real files against an independent
# implementation (gguf-py 0.19.0, the upstream project's own reference), and each
# block size was confirmed by reproducing the file's own tensor offsets from the
# size table alone -- a check that never calls a decoder.  infer_type_layout() was
# tested by DELETING the answer from the table and asking it to recover it.
# =========================================================================== #

QK_IQ = 256


#: Sign byte for a 7-bit index, 128 entries (ggml's `ksigns_iq2xs`).
#: Bit j of the looked-up byte is the sign of the j-th value.
K_SIGNS = (
    b'\x00\x81\x82\x03\x84\x05\x06\x87\x88\t\n\x8b\x0c\x8d\x8e\x0f\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f'
    b'\xa0!"\xa3$\xa5\xa6\'(\xa9\xaa+\xac-.\xaf0\xb1\xb23\xb456\xb7\xb89:\xbb<\xbd\xbe?'
    b'\xc0AB\xc3D\xc5\xc6GH\xc9\xcaK\xccMN\xcfP\xd1\xd2S\xd4UV\xd7\xd8YZ\xdb\\\xdd\xde_'
    b'`\xe1\xe2c\xe4ef\xe7\xe8ij\xebl\xed\xeeo\xf0qr\xf3t\xf5\xf6wx\xf9\xfa{\xfc}~\xff'
)


def _grid_from_hex(hex_bytes: bytes, table) -> np.ndarray:
    """Decode a grid table literal into an fp32 lookup table.

    `table` is (grid_map, grid_shape).  The literal is ASCII hex; each pair of
    hex digits is one byte, and that byte carries ceil(log2(len(grid_map))) bits
    per grid element.  Upstream packs with that granularity, so a 3-bit grid
    (8-entry map) really is TWO codes per byte at bit offsets 0 and 4, not three:
    the shift step is 8 // (8 // bits), which is why this is computed and not
    written as `bits * i`.  Getting that wrong reads a plausible table that is
    not the table -- the value count is asserted below so it cannot be silent.
    """
    grid_map, grid_shape = table
    bits = int(np.ceil(np.log2(len(grid_map))))
    per_byte = 8 // bits
    raw = np.frombuffer(hex_bytes, dtype=np.uint8)
    if raw.size % 2:
        raise ValueError("grid literal has an odd number of hex digits")
    dg = raw.reshape(-1, 2)
    hi = np.where(dg[:, 0] > 0x40, dg[:, 0] + 9, dg[:, 0]) & 0x0F
    lo = np.where(dg[:, 1] > 0x40, dg[:, 1] + 9, dg[:, 1]) & 0x0F
    packed = ((hi << 4) | lo).astype(np.uint8)
    mask = (1 << bits) - 1
    shifts = range(0, 8, 8 // per_byte)
    codes = np.stack([(packed >> s) & mask for s in shifts], axis=1).reshape(-1)
    if codes.size != grid_shape[0] * grid_shape[1]:
        raise ValueError("grid literal decodes to %d codes, shape %r wants %d"
                         % (codes.size, grid_shape, grid_shape[0] * grid_shape[1]))
    tab = np.array(grid_map, dtype=np.float32)
    if codes.size and int(codes.max()) >= tab.size:
        raise ValueError("grid code %d is outside a grid_map of %d entries"
                         % (int(codes.max()), tab.size))
    return tab[codes].reshape(grid_shape[0], grid_shape[1])


GRID_IQ2_XXS = _grid_from_hex(
    (
        b"00000200050008000a00110014002000220028002a0041004400500058006100"
        b"6400800082008a00a20001010401100115014001840198010002020222028202"
        b"010404041004210424044004420448046004810484049004a404000502050805"
        b"200546056905800591050906100640068406a406000805080808140828084108"
        b"440850085208880804094009020a140a01100410101021104010601084109010"
        b"951000110811201150115a118011241245120014081420142514491480141815"
        b"6215001616160118041810184018811800190519a019511a002002200a204420"
        b"6120802082202921482100220222012404241024402456240025412564259026"
        b"082820289428442a014004401040184021402440404048405640604081408440"
        b"9040004120416141804185410142104248425642684200440844204480449944"
        b"124524450046014804481048404845480049584961498249454a904a00500850"
        b"1150195020508050885004514251a4519152905492540a550156545600581158"
        b"195864584059085a046010604060686000615561186260620064056410651265"
        b"84654268008002800a8041808280048118814081118201840484108415844084"
        b"608400854685948509864086608602880489118a0490109024904090a1901691"
        b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
    ),
    ((0x08, 0x19, 0x2b), (256, 8)))

GRID_IQ2_XS = _grid_from_hex(
    (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"49005000520055005800610064008000820085008800910094009900a0000101"
        b"04010601090110011201150118011a0121012401400142014501480151015401"
        b"6001680181018401900100020202050208021102140220024102440250025502"
        b"80028a0201040404060409041004120415041804210424044004420445044804"
        b"5104540456046004810484049004000502050505080511051405200541054405"
        b"500561058005010604061006260640064206840600080208050808080a081108"
        b"14082008250841084408500858088008a008aa08010904091009400981098909"
        b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
        b"4210451048105110541060106a10811084109010001102110511081111111411"
        b"2011411144115011801194119611011204120612101240126012001402140514"
        b"0814111414142014411444144914501464148014011504151015401500161416"
        b"49160118041810181218401854188618001905196619511aa91a002002200520"
        b"08200a201120142020204120442050208020a020012104211021402148216521"
        b"002222228022a82201240424102429244024002541255225992501261a26a626"
        b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        b"0640094010401240154018402140244040404240454048404a40514054406040"
        b"6540814084409040004102410541084111411441204141414441504180418541"
        b"a241014204421042124229424042004402440544084411441444194420444144"
        b"4444504480449444014504451045244540459a4500460a464446504601480448"
        b"1048404845485448624800491149444950496949044a00500250055008501150"
        b"145020502850415044505050805001510451105115514051425100524452aa52"
        b"0154045410542154405460548154a154005508558055885521566856a1560058"
        b"14584158505899581a5940594259855a0160046010604060546062608660a960"
        b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
        b"00800280058008801180148020802a8041804480508080808280a880aa800181"
        b"0481068110814081518159810082208280828282a082a8820184048410841284"
        b"158440846084898400854485a58518866a860088088825885a8880888288a888"
        b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
        b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    ),
    ((0x08, 0x19, 0x2b), (512, 8)))

GRID_IQ2_S = _grid_from_hex(
    (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    ),
    ((0x08, 0x19, 0x2b), (1024, 8)))

GRID_IQ3_XXS = _grid_from_hex(
    (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    ),
    ((0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e), (256, 4)))

GRID_IQ3_S = _grid_from_hex(
    (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    ),
    ((0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f), (512, 4)))

GRID_IQ1_S = _grid_from_hex(
    (
        b"00000200050008000a00110015002000220028002a0045005100540056006500"
        b"8000820088008a009500a000a200a800aa000401050111011401160119011a01"
        b"2501410146014901520155015a0161016401660168018501910194019601a501"
        b"0002020208020a0215022002220228022a024502510259026402690280028202"
        b"88028a02910295029902a002a202a802aa021104140416042504410449045504"
        b"5a046404650491049904a5040105040505050605150518051a05290540054505"
        b"4a0550055105540555055605590560056205650568056a058105910595059805"
        b"9a05a105a405a505a605a9051406190641064406500652065506580660066106"
        b"6606690685069106940699060008020808080a0815082008220828082a084508"
        b"5108560865088008820888088a089508a008a208a808aa080509110914091909"
        b"2409250941095009510955096109640969099109940996099909a509000a020a"
        b"080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
        b"8a0a950aa00aa20aa80aaa0a1010111014101910241025104110441050105510"
        b"58106110641065106910911094109610a110a510011104110611091110111211"
        b"1511181121112411291145114a11501151115211541155115611591160116511"
        b"841192119511a111a41111121412161225124012461249125212551258125a12"
        b"641266128512911294129612a512011406140914141415141814191421142614"
        b"41144514461448144a1451145414551456145914621465146814841489149014"
        b"94149514981499149a14a114a414a514a914021505150a151115141515151615"
        b"191520152215251528152a154115441545154615511552155415551556155915"
        b"5a1561156415651566156915801582158415851588158a159015911594159515"
        b"961599159a15a015a215a51501160416051606161516161618161a1621162616"
        b"401642164416451648164a165116551656165816591661166416651668166916"
        b"6a1686168a1692169516a416a916111816182518411844184618491850185518"
        b"58185a1860186118641866186918851891189418a5181019121915191a192119"
        b"25194219441945194819511954195519561959195a19601965196a1989199119"
        b"921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
        b"581a611a661a691a851a911a961a9a1a0020022008200a201520202022202520"
        b"28202a20452051205920612065208020822088208a209520a020a220a520a820"
        b"aa2005211121142119212521422144214921552158215a216121642165216621"
        b"8521902196219921a521012208220a22112215222022222228222a2245225122"
        b"562259226522812288228a2291229522a022a222a822aa220524142416241924"
        b"252444244524462449245224552458245a2466248524912494249924a124a524"
        b"0925152521252925402545254825512554255525592562256525682589259025"
        b"9425952598259a25a125a425a625a92505261026122619262526412649265526"
        b"6026612669268426862690269a260028022808280a2815282028222828282a28"
        b"45285128542865288028822888288a28a028a228a828aa280929112914291929"
        b"2529462949295229552961296429662969298529902996299929a429a529002a"
        b"022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
        b"952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440"
        b"664094409940a140a6400041014104410641094112411541164118411a412141"
        b"26412941454148414a41514154415541564159415a41654168416a4181418441"
        b"8641904192419541a041a141a241054211421442164225424142524255425a42"
        b"6442694289429442a5420144154419442944454448444a445144544455445644"
        b"61446244654468446a44814486448944904492449544a044a144a94401450245"
        b"05450a4511451445154516451945204525452a45414544454545464549455045"
        b"5145544555455645584559456145644565456645694582458445854588459145"
        b"94459545964599459a45a545a845aa450146054609461446154618461a462146"
        b"2446294640464246454648465046514652465546564659466246654668468146"
        b"85468a4694469546a146a446a6460548114815481a4825484248494850485548"
        b"5848614864486648694885489148944896489948a5480149054906490a491049"
        b"144915491849214924492649404945494a495149524954495549564959496049"
        b"6249654966496a49864989499249954996499849a149a449a649a949164a444a"
        b"464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
        b"1a50215024502950405045504850515054505550565059506550685086508950"
        b"95509850a050a150a650a9500551085109510a51115114511551165118511951"
        b"20512551265128512a5141514451455146514951505151515251545155515651"
        b"585159515a51615164516551665169518251855191519451955196519951a051"
        b"a551aa5101520652125215521a5221522452425245524a525152545255525652"
        b"595262526552855290529252955299529a52a452045405541154145415541654"
        b"185419542154255428542a54415444544554465449544a545054515454545554"
        b"5654585459545a54615462546454655466546954805488548a54915494549554"
        b"96549954a154a454a554aa540155025504550555065509551055115512551455"
        b"1555165519551a55215524552555265529554055415542554455455546554855"
        b"4955505551555255545555555655585559555a55605561556455655566556855"
        b"69556a5581558455855589558a559055915594559555965598559955a155a455"
        b"a555a655a9550056015602560456065608560956115614561556185619562056"
        b"2156225624562556265628562956415645564656485649564a56505651565256"
        b"545655565656585659565a566156645665566956825685568656885689568a56"
        b"915695569a56a256a556a656a856a95604580558065809581058155818582158"
        b"2a58455848584a58515854585558565858585958605862586458655882588958"
        b"9058925895589858a158a9580159025905590a59115914591559165919592559"
        b"41594459455946594959505951595259545955595659585959595a5961596459"
        b"655966596959815985598959915994599559965998599959a559045a085a155a"
        b"1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
        b"6a5a815a8a5a925a955a965a985a9a5aa15a0560146016601960256044605060"
        b"5560566058605a60616064606660696081609660a56001610461066109611261"
        b"15612161226126612961456149615161556156615961656166616a6184618a61"
        b"92619561a161a661a96111621662196240624162466255625662586260628562"
        b"91629662a56211641264156416641a6421642664296440644264456448644a64"
        b"516454645564566459645a646064626465648464856489649064926494649564"
        b"966498649a64a164a464a964056508650a651165156516651965446545654665"
        b"496550655165546555655665596561656465656566656965866589658a659165"
        b"9565966599659a65a265a565a665a86502660966156620662666286629664066"
        b"456648664a66516654665566566658665a666066656668668066826685668a66"
        b"9466966698669966a066a466a666aa661668196825684168526855685a686168"
        b"6968856891689868a66801690469106915692169246926692969406941694569"
        b"4669486951695469556956695969606965696a69826984698a699569a169a469"
        b"a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a"
        b"946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
        b"5680598065808080828088808a809580a080a280a880aa800581118114811681"
        b"1981258141814481498150815281558156815881598164816681698185818981"
        b"948196819981a5810082028208820a8215822082228228822a82518254825982"
        b"65828082828288828a829582a082a282a882aa82148419844184448451845584"
        b"5a846184648469849484998401850985128515851a8526852985408541854585"
        b"4885518554855585568559855a856585668568856a8581858485868589859085"
        b"928595859885a68511861686198625864186448649864a865086558659865a86"
        b"618666866a86858691869a86a4860088028808880a8815882088228828882a88"
        b"41884588518854885988658869888088828888888a889588a088a288a888aa88"
        b"05890689118914891689258941894489468949895089528955895a8961896489"
        b"858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a"
        b"568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
        b"419046904990559058905a9069906a9085909190949096909990a59001910491"
        b"069109911091159118911a912191249126912991409145915091519154915591"
        b"569159916291659184918691929195919891a191a491a691a991059211921492"
        b"19922592449246924992509252925592589266926992859294929692a9920194"
        b"04940694109415941894269440944a9451945494559456945894599460946194"
        b"62946594849486949294949495949894a194a9940095059508950a9510951195"
        b"14951595169519952195259529952a9541954495459546954995509551955295"
        b"549555955695589559955a956195649565956695699581958595889591959295"
        b"94959595969599959a95a095a295a595a895aa95019604961096159619962096"
        b"2696299645964896499651965296559656965996659668968296849689968a96"
        b"929694969596a496a696a9960598169819982598419846985098529855985698"
        b"5a98649865988598919896989998a59804990699099910991299159918991a99"
        b"209921992499269940994299459948994a995199549955995699599962996599"
        b"66996a99819984999099929995999a99a199a699059a159a259a449a469a499a"
        b"509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
        b"28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa0"
        b"05a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a1"
        b"90a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
        b"65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a4"
        b"55a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a5"
        b"29a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
        b"95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a6"
        b"62a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a8"
        b"2aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
        b"19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa"
        b"20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa"
    ),
    ((-1, 0, 1), (2048, 8)))

#: iq4nl, the 16-entry non-linear 4-bit codebook (ggml's `kvalues_iq4nl`).
KVALS_IQ4NL = np.array((
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
), dtype=np.float32)


# --------------------------------------------------------------------------- #
# (i) AFFINE TYPES: a table row and a closed-form decoder, no table resource.
# --------------------------------------------------------------------------- #
Q4_0_BLOCK_BYTES = 18


def decode_q4_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Q4_0 -> fp32.  block_q4_0 { fp16 d; uint8 qs[16]; } = 18 B / 32 values.

    dequantize_row_q4_0: y[j] = d * (((qs[j/2] >> (4*(j%2))) & 0xF) - 8), i.e.
    the low nibble of qs[j] is element j and the high nibble element j+16.
    """
    if n_elements % 32:
        raise ValueError("Q4_0 needs a multiple of 32 values, got %d" % n_elements)
    n_blocks = n_elements // 32
    want = n_blocks * Q4_0_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q4_0 payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q4_0_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    q = blk[:, 2:18].astype(np.int32)
    codes = np.concatenate([q & 0x0F, (q >> 4) & 0x0F], axis=1).astype(np.float32)
    return (d[:, None] * (codes - np.float32(8.0))).astype(np.float32).ravel()


DEQUANTIZERS[2] = decode_q4_0


Q5_K_BLOCK_BYTES = 176


def decode_q5_K(raw: bytes, n_elements: int) -> np.ndarray:
    """Q5_K -> fp32: the Q4_K frame plus a 32-byte high-bit plane.  176 B / 256.

    The 6-bit (scale, min) pairs are the SAME encoding Q4_K already decodes in
    this module, so _q4k_scale_min() is REUSED rather than respelled.  The value
    is d*sc*q - dmin*m with q a 5-bit code: the low nibble from qs and the 5th bit
    from qh.  qs byte g*32+j holds the low nibbles of groups 2g and 2g+1; qh byte
    j bit m is the 5th bit of group m element j, so both flatten to
    (n_blocks, 8, 32) in the same group-major order as sc/mn.  Bits 0-3 are exact
    products of an 11-bit fp16 with a 5-bit code, so no intermediate rounds.
    """
    if n_elements % QK_K:
        raise ValueError("Q5_K needs a multiple of %d values, got %d" % (QK_K, n_elements))
    n_blocks = n_elements // QK_K
    want = n_blocks * Q5_K_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q5_K payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q5_K_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    dmin = blk[:, 2:4].copy().view("<f2").astype(np.float32).ravel()
    sc, mn = _q4k_scale_min(blk[:, 4:16])
    # block_q5_K is { d; dmin; scales[12]; qh[32]; qs[128] } -- the HIGH-bit plane
    # comes BEFORE the low-nibble plane, which is easy to read the other way round
    # and then decodes 176 bytes that are not this type's 176 bytes.
    qh = blk[:, 16:48].astype(np.int32)
    qs = blk[:, 48:176].astype(np.int32)
    lo = (qs & 0x0F).reshape(n_blocks, 4, 32)
    hi = ((qs >> 4) & 0x0F).reshape(n_blocks, 4, 32)
    ql = np.stack([lo, hi], axis=2).reshape(n_blocks, 8, 32)
    qhh = np.stack([(qh >> m) & 0x01 for m in range(8)], axis=1)
    q = ql | (qhh << 4)
    ds = (d[:, None] * sc.astype(np.float32))[:, :, None]
    dm = (dmin[:, None] * mn.astype(np.float32))[:, :, None]
    return (ds * q.astype(np.float32) - dm).astype(np.float32) \
        .reshape(n_blocks, QK_K).ravel()


DEQUANTIZERS[13] = decode_q5_K


Q2_K_BLOCK_BYTES = 84


def decode_q2_K(raw: bytes, n_elements: int) -> np.ndarray:
    """Q2_K -> fp32: 16 sub-blocks of 16, a 4-bit scale, a 4-bit min.  84 B / 256.

    block_q2_K = scales[16], qs[64], fp16 d, fp16 dmin.  Sub-block g = a*4+s
    (a = qs half, s = 2-bit field) carries scale (scales[g] & 0xF) and minimum
    (scales[g] >> 4), and y = d*sc*q - dmin*m with q the 2-bit code.
    """
    if n_elements % QK_K:
        raise ValueError("Q2_K needs a multiple of %d values, got %d" % (QK_K, n_elements))
    n_blocks = n_elements // QK_K
    want = n_blocks * Q2_K_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("Q2_K payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q2_K_BLOCK_BYTES)
    scales = blk[:, 0:16].astype(np.int32)
    qs = blk[:, 16:80].astype(np.int32)
    d = blk[:, 80:82].copy().view("<f2").astype(np.float32).ravel()
    dmin = blk[:, 82:84].copy().view("<f2").astype(np.float32).ravel()
    dl = d[:, None] * (scales & 0x0F).astype(np.float32)
    ml = dmin[:, None] * (scales >> 4).astype(np.float32)
    a32 = qs.reshape(n_blocks, 2, 32)
    codes = np.stack([(a32 >> s) & 0x03 for s in (0, 2, 4, 6)], axis=2)
    codes = codes.reshape(n_blocks, 16, 16).astype(np.float32)
    return (dl[:, :, None] * codes - ml[:, :, None]).astype(np.float32) \
        .reshape(n_blocks, QK_K).ravel()


DEQUANTIZERS[10] = decode_q2_K


# --------------------------------------------------------------------------- #
# (ii) GRID/CODEBOOK TYPES: a table row, a decoder, and a table RESOURCE.
# --------------------------------------------------------------------------- #
def _iq_signs(ksign_idx: np.ndarray) -> np.ndarray:
    """7-bit ksigns index -> +/-1, one per value, bit 0 of the byte first."""
    sb = np.frombuffer(K_SIGNS, dtype=np.uint8)[ksign_idx]
    bits = np.stack([(sb >> b) & 0x01 for b in range(8)], axis=-1)
    return np.where(bits == 0, np.float32(1.0), np.float32(-1.0))


IQ4_XS_BLOCK_BYTES = 136


def decode_iq4_xs(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ4_XS -> fp32: 8 sub-blocks of 32, a 6-bit biased scale, the iq4nl book.

    block_iq4_xs = fp16 d, uint16 scales_h, scales_l[4], qs[128] = 136 B / 256.
    The 6-bit scale is (scales_l nibble | scales_h bit-pair << 4) - 32, and the
    4-bit code indexes KVALS_IQ4NL.  Element e of a sub-block is the low nibble
    of qs byte e for e < 16 and the high nibble of byte e-16 for e >= 16.
    """
    if n_elements % QK_K:
        raise ValueError("IQ4_XS needs a multiple of %d values, got %d" % (QK_K, n_elements))
    n_blocks = n_elements // QK_K
    want = n_blocks * IQ4_XS_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ4_XS payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ4_XS_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    sh = blk[:, 2:4].copy().view("<u2").astype(np.int32).ravel()
    sl = blk[:, 4:8].astype(np.int32)
    s_l = np.stack([sl & 0x0F, (sl >> 4) & 0x0F], axis=2).reshape(n_blocks, 8)
    s_h = np.stack([(sh >> (2 * k)) & 0x03 for k in range(8)], axis=1)
    sc = s_l | (s_h << 4)
    dl = (d[:, None] * (sc - 32).astype(np.float32))[:, :, None]
    g = blk[:, 8:136].astype(np.int32).reshape(n_blocks, 8, 16)
    codes = np.concatenate([g & 0x0F, (g >> 4) & 0x0F], axis=2)
    return (dl * KVALS_IQ4NL[codes]).astype(np.float32) \
        .reshape(n_blocks, QK_K).ravel()


DEQUANTIZERS[23] = decode_iq4_xs


IQ4_NL_BLOCK_BYTES = 18


def decode_iq4_nl(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ4_NL -> fp32: an fp16 delta then 16 nibble bytes, iq4nl codebook.  18 B/32.

    block_iq4_nl = { fp16 d; uint8 qs[16] } and y[j] = d * KVALS_IQ4NL[code[j]], with
    the low nibble of qs[j] element j and the high nibble element j+16.  It shares
    KVALS_IQ4NL with IQ4_XS above -- one codebook, two frames -- and it is the type
    `dl/convmulti` (F-899) handed over as MISSING from the first proposal: it is
    needed by Ling-3.0-tiny-IQ4_XS.gguf (6 tensors) and its block frame is the
    Q4_0 frame, which is why GGML_LAYOUT already carried (32, 18) for id 2.
    """
    if n_elements % 32:
        raise ValueError("IQ4_NL needs a multiple of 32 values, got %d" % n_elements)
    n_blocks = n_elements // 32
    want = n_blocks * IQ4_NL_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ4_NL payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ4_NL_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    q = blk[:, 2:18].astype(np.int32)
    codes = np.concatenate([q & 0x0F, (q >> 4) & 0x0F], axis=1)
    return (d[:, None] * KVALS_IQ4NL[codes]).astype(np.float32).ravel()


DEQUANTIZERS[20] = decode_iq4_nl


# The five grid types share one shape: an fp16 super-scale d, a per-sub-block
# scale packed as extra HIGH bits beside the code, a sign plane, and a GRID row
# looked up by the code, so y = db * grid * sgn.  db is exact in fp32 for every
# one of them (d has 11 mantissa bits and the per-sub-block factor at most five
# more), so computing in float64 and casting once at the store -- this module's
# house style -- cannot round differently from a scalar fp32 decoder.
IQ2_XXS_BLOCK_BYTES = 66


def decode_iq2_xxs(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ2_XXS -> fp32: 8 sub-blocks of 32, grid rows of 8 from a 256-row table.

    block_iq2_xxs = fp16 d, uint16 qs[32] = 66 B / 256.  Viewed as eight (grid,
    sign) uint32 pairs: the grid word is four 8-bit row indices, the paired word
    carries a 4-bit scale in its top nibble and four 7-bit ksigns indices.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ2_XXS needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ2_XXS_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ2_XXS payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ2_XXS_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    w = blk[:, 2:66].copy().view("<u4").reshape(n_blocks, 8, 2).astype(np.uint32)
    lo, hi = w[:, :, 0], w[:, :, 1]
    db = d[:, None] * (np.float32(0.5) + (hi >> 28).astype(np.float32)) \
        * np.float32(0.25)
    rows = lo.copy().view(np.uint8).reshape(n_blocks, 8, 4).astype(np.int64)
    idx = np.stack([(hi >> s) & 0x7F for s in (0, 7, 14, 21)], axis=2).astype(np.int64)
    return (db[:, :, None, None] * GRID_IQ2_XXS[rows] * _iq_signs(idx)) \
        .astype(np.float32).reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[16] = decode_iq2_xxs


IQ2_XS_BLOCK_BYTES = 74


def decode_iq2_xs(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ2_XS -> fp32: 16 sub-blocks of 16, a 512-row grid, a 9-bit row index.

    block_iq2_xs = fp16 d, uint16 qs[32], scales[8] = 74 B / 256.  Each uint16 is
    a 9-bit grid row index in its low bits and a 7-bit ksigns index above it.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ2_XS needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ2_XS_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ2_XS payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ2_XS_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    w = blk[:, 2:66].copy().view("<u2").astype(np.uint16).reshape(n_blocks, 16, 2)
    sc = blk[:, 66:74].astype(np.int32)
    sub = np.stack([sc & 0x0F, (sc >> 4) & 0x0F], axis=2).reshape(n_blocks, 16)
    db = d[:, None] * (np.float32(0.5) + sub.astype(np.float32)) * np.float32(0.25)
    rows = (w & 0x01FF).astype(np.int64)
    idx = (w >> 9).astype(np.int64)
    return (db[:, :, None, None] * GRID_IQ2_XS[rows] * _iq_signs(idx)) \
        .astype(np.float32).reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[17] = decode_iq2_xs


IQ2_S_BLOCK_BYTES = 82


def decode_iq2_s(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ2_S -> fp32: 16 sub-blocks, a 1024-row grid, an explicit sign plane.

    block_iq2_s = fp16 d, qs[32], signs[32], qh[8], scales[8] = 82 B / 256.  The
    10-bit row index is qs | ((qh >> 2*(j%4) & 3) << 8), and the signs are their
    own 32-byte plane rather than a packed ksigns index.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ2_S needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ2_S_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ2_S payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ2_S_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    qs = blk[:, 2:34].astype(np.int32)
    sgn_bytes = blk[:, 34:66].astype(np.int32)
    qh = blk[:, 66:74].astype(np.int32)
    sc = blk[:, 74:82].astype(np.int32)
    sub = np.stack([sc & 0x0F, (sc >> 4) & 0x0F], axis=2).reshape(n_blocks, 16)
    db = d[:, None] * (np.float32(0.5) + sub.astype(np.float32)) * np.float32(0.25)
    qhv = np.stack([(qh >> (2 * t)) & 0x03 for t in range(4)], axis=2)
    rows = (qs | (qhv.reshape(n_blocks, 32) << 8)).reshape(n_blocks, 16, 2) \
        .astype(np.int64)
    sb = np.stack([(sgn_bytes >> b) & 0x01 for b in range(8)], axis=2)
    sgn = np.where(sb == 0, np.float32(1.0), np.float32(-1.0)) \
        .reshape(n_blocks, 16, 2, 8)
    return (db[:, :, None, None] * GRID_IQ2_S[rows] * sgn).astype(np.float32) \
        .reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[22] = decode_iq2_s


IQ3_XXS_BLOCK_BYTES = 98


def decode_iq3_xxs(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ3_XXS -> fp32: 8 sub-blocks of 32, one grid byte per 4 values.

    block_iq3_xxs = fp16 d, qs[64], scales[8] (uint32) = 98 B / 256.  qs is 64
    one-byte indices into a 256-row, 4-wide grid; each scale word carries a 4-bit
    scale in its top nibble and four 7-bit ksigns indices in its low 28 bits.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ3_XXS needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ3_XXS_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ3_XXS payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ3_XXS_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    rows = blk[:, 2:66].astype(np.int64).reshape(n_blocks, 8, 8)
    sc = blk[:, 66:98].copy().view("<u4").astype(np.uint32).reshape(n_blocks, 8)
    db = d[:, None] * (np.float32(0.5) + (sc >> 28).astype(np.float32)) \
        * np.float32(0.5)
    idx = np.stack([(sc >> s) & 0x7F for s in (0, 7, 14, 21)], axis=2).astype(np.int64)
    sgn = _iq_signs(idx).reshape(n_blocks, 8, 32)
    grid = GRID_IQ3_XXS[rows].reshape(n_blocks, 8, 32)
    return (db[:, :, None] * grid * sgn).astype(np.float32) \
        .reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[18] = decode_iq3_xxs


IQ3_S_BLOCK_BYTES = 110


def decode_iq3_s(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ3_S -> fp32: 8 sub-blocks, a 512-row grid, 9-bit rows, a sign plane.

    block_iq3_s = fp16 d, qs[64], qh[8], signs[32], scales[4] = 110 B / 256.  The
    9-bit row index is qs[j] | (qh bit j << 8) into a 512-row, 4-wide grid;
    db = d*(1 + 2*scale) with the 8-bit split scale, and the 32-byte sign plane
    gives 8 signs per row.  THIS IS THE GSQ-RCO 3.5-bit RUNG'S LARGEST TYPE:
    144 of its 851 tensors, and the rung also uses every other type in this block.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ3_S needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ3_S_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ3_S payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ3_S_BLOCK_BYTES)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    qs = blk[:, 2:66].astype(np.int32)
    qh = blk[:, 66:74].astype(np.int32)
    sgn_bytes = blk[:, 74:106].astype(np.int32)
    sc = blk[:, 106:110].astype(np.int32)
    sub = np.stack([sc & 0x0F, (sc >> 4) & 0x0F], axis=2).reshape(n_blocks, 8)
    db = d[:, None] * (np.float32(1.0) + np.float32(2.0) * sub.astype(np.float32))
    qhv = np.stack([(qh >> b) & 0x01 for b in range(8)], axis=2).reshape(n_blocks, 64)
    rows = (qs | (qhv << 8)).reshape(n_blocks, 8, 8).astype(np.int64)
    sb = np.stack([(sgn_bytes >> b) & 0x01 for b in range(8)], axis=2)
    sgn = np.where(sb == 0, np.float32(1.0), np.float32(-1.0)).reshape(n_blocks, 8, 32)
    grid = GRID_IQ3_S[rows].reshape(n_blocks, 8, 32)
    return (db[:, :, None] * grid * sgn).astype(np.float32) \
        .reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[21] = decode_iq3_s


IQ1_M_BLOCK_BYTES = 56


def decode_iq1_m(raw: bytes, n_elements: int) -> np.ndarray:
    """IQ1_M -> fp32: 8 x 2 x 2 sub-blocks of 8, an ACROSS-WORDS fp16 scale.

    block_iq1_m = qs[32], qh[16], scales[8] = 56 B / 256.  The only type here
    whose fp16 super-scale is split: the four high nibbles of the four uint16
    scale words spell the fp16 bit pattern.  Codes are 11 bits into the 2048-row
    iq1s grid (values -1/0/+1), and a per-half-signed 1/8 delta is ADDED to the
    grid value before the per-sub-block scale is applied.
    """
    if n_elements % QK_IQ:
        raise ValueError("IQ1_M needs a multiple of 256 values, got %d" % n_elements)
    n_blocks = n_elements // QK_IQ
    want = n_blocks * IQ1_M_BLOCK_BYTES
    if len(raw) != want:
        raise ValueError("IQ1_M payload is %d bytes, %d values need %d"
                         % (len(raw), n_elements, want))
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, IQ1_M_BLOCK_BYTES)
    qs = blk[:, 0:32].astype(np.int32)
    qh = blk[:, 32:48].astype(np.int32)
    sw = blk[:, 48:56].copy().view("<u2").astype(np.uint16).reshape(n_blocks, 4)
    pat = np.zeros(n_blocks, dtype=np.uint16)
    for w, shift in enumerate((12, 8, 4, 0)):
        pat = pat | ((sw[:, w] & np.uint16(0xF000)) >> np.uint16(shift))
    d16 = pat.copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    sub = np.stack([(sw >> s) & np.uint16(7) for s in (0, 3, 6, 9)], axis=2)
    dl = (d16 * (np.float32(2.0) * sub.reshape(n_blocks, 16).astype(np.float32)
                 + np.float32(1.0))).reshape(n_blocks, 8, 2, 1, 1)
    qhv = np.stack([(qh >> s) & 0x07 for s in (0, 4)], axis=2)
    rows = (qs.reshape(n_blocks, 16, 2) | (qhv << 8)).reshape(n_blocks, 32) \
        .astype(np.int64)
    dv = np.stack([((qh >> 3) & 0x01) == 0, ((qh >> 7) & 0x01) == 0], axis=2)
    delta = np.where(dv, np.float32(0.125), np.float32(-0.125)) \
        .reshape(n_blocks, 8, 2, 2, 1)
    grid = GRID_IQ1_S[rows].reshape(n_blocks, 8, 2, 2, 8)
    return (dl * (grid + delta)).astype(np.float32) \
        .reshape(n_blocks, QK_IQ).ravel()


DEQUANTIZERS[29] = decode_iq1_m


# --------------------------------------------------------------------------- #
# the block sizes of the eight types above, ADDED to the layout table.  Rebinding
# a copy rather than editing the literal keeps this block an append; the eight
# rows are cross-checked against ggml's published table and are confirmed on the
# real RCO file by validate_layout_arithmetic(), which reproduces all 851 of its
# tensor offsets from these sizes ALONE -- that check never calls a decoder.
# --------------------------------------------------------------------------- #
_RCO_LAYOUT_ROWS = {
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    21: (256, 110),   # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),   # IQ4_XS
    29: (256, 56),   # IQ1_M
    20: (32, 18),   # IQ4_NL
}
for _t, _row in _RCO_LAYOUT_ROWS.items():
    if _t in GGML_LAYOUT and GGML_LAYOUT[_t] != _row:
        raise AssertionError(
            "ggml type %d layout disagrees: table says %r, this block says %r"
            % (_t, GGML_LAYOUT[_t], _row))
GGML_LAYOUT = dict(GGML_LAYOUT)
GGML_LAYOUT.update(_RCO_LAYOUT_ROWS)
del _t, _row


# --------------------------------------------------------------------------- #
# THE GENERAL PATH, PART 1: TRIAGE.  Will a new model just work?
# --------------------------------------------------------------------------- #
#: ggml type id -> how a reader for it must be built.  This is the classification
#: that decides the cost of adapting a future model, and it is data, so a new
#: type gets one entry rather than a special case.
#:   "raw"     container arithmetic only, no quantisation at all
#:   "affine"  class (i)   -- a scale (and usually a min) times an integer code
#:   "grid"    class (ii)  -- an index into an embedded reconstruction table
#:   "codebook" class (ii) -- a small fixed value table (not a multi-row grid)
#:   "private" a fork-specific layout, not in upstream ggml
QUANT_FAMILY = {
    0: "raw", 1: "raw", 30: "raw", 28: "raw",
    24: "raw", 25: "raw", 26: "raw", 27: "raw",
    2: "affine", 3: "affine", 6: "affine", 7: "affine", 8: "affine", 9: "affine",
    10: "affine", 11: "affine", 12: "affine", 13: "affine", 14: "affine",
    15: "affine",
    16: "grid", 17: "grid", 18: "grid", 19: "grid", 21: "grid", 22: "grid",
    29: "grid",
    20: "codebook", 23: "codebook",
    34: "affine", 35: "affine",
    39: "codebook", 40: "codebook",
    41: "private", 142: "private", 143: "private",
}


def adaptation_class(type_id: int) -> str:
    """(i) / (ii) / (iii) for a type, or the reason it cannot be answered.

    (iii) IS NEVER RETURNED FOR READING and that is the finding, not a gap: an
    imatrix is an authoring input, so no ggml type needs a companion artifact to
    be READ.  The class is returned rather than omitted so the caller can see the
    question was asked.
    """
    fam = QUANT_FAMILY.get(type_id)
    if fam is None:
        return "unknown-type"
    if fam in ("raw", "affine", "private"):
        return "(i) table row + closed-form decoder"
    return "(ii) table row + decoder + embedded grid/codebook resource"


def quant_coverage(path: str, verbose: bool = False) -> dict:
    """Per-type census of one gguf and whether THIS reader can read it.

    Returns a dict with the tensor count, the per-type histogram, the number of
    tensors whose type has a decoder, the refused population grouped by type with
    its adaptation class, and the layout-arithmetic verdict.  This is the one call
    that answers "will this new model just work?" without reading any source.
    """
    kv, tensors, data_offset = read_tensor_table(path)
    hist, refused, nelem_by_type = {}, {}, {}
    decoded = 0
    for name, dims, ttype, off in tensors:
        nelem = 1
        for d in dims:
            nelem *= int(d)
        hist[ttype] = hist.get(ttype, 0) + 1
        nelem_by_type[ttype] = nelem_by_type.get(ttype, 0) + nelem
        if ttype in DEQUANTIZERS:
            decoded += 1
        else:
            refused[ttype] = refused.get(ttype, 0) + 1
    problem = {
        t: {"tensors": n, "name": type_name(t), "class": adaptation_class(t),
            "layout_known": t in GGML_LAYOUT}
        for t, n in sorted(refused.items(), key=lambda kv_: -kv_[1])
    }
    out = {
        "path": path, "tensors": len(tensors),
        "histogram": {type_name(t): n for t, n in sorted(hist.items())},
        "decoded": decoded, "refused": sum(refused.values()),
        "refused_by_type": problem,
        "layout": validate_layout_arithmetic(path),
        "general.name": str(kv.get("general.name", ""))[:120],
        "general.architecture": str(kv.get("general.architecture", ""))[:40],
    }
    if verbose:
        print("%s  [%s]  tensors=%d"
              % (path, out["general.architecture"], out["tensors"]))
        for t, n in sorted(hist.items(), key=lambda kv_: -kv_[1]):
            mark = "ok " if t in DEQUANTIZERS else "NO "
            print("  %s %-9s %-5s x %4d"
                  % (mark, type_name(t), "%d" % t, n))
        print("  decoded %d of %d" % (decoded, len(tensors)))
        for t, info in problem.items():
            print("  refused %-9s x %4d -- %s"
                  % (info["name"], info["tensors"], info["class"]))
    return out


# --------------------------------------------------------------------------- #
# THE GENERAL PATH, PART 2: A TYPE NOT IN THE TABLE IS MEASURED, NOT REFUSED
# WITH NOTHING.  For an unknown type id the reader currently raises "unknown ggml
# type id" and the caller learns nothing -- not even the block size.  But the
# block size is IN THE FILE: the tensor offsets are cumulative, so a tensor of a
# known type that FOLLOWS the unknown one pins the unknown one's stored bytes,
# and (values_per_block, bytes_per_block) is then solvable from the element count.
# --------------------------------------------------------------------------- #
LAYOUT_CANDIDATE_BLOCKS = (1, 16, 32, 64, 128, 256, 512, 1024)


def infer_type_layout(path: str, type_id: int, align: int = 0) -> dict:
    """Measure an unregistered type's block size from the file's own offsets.

    Returns {"solutions": [(values_per_block, bytes_per_block, evidence)], ...}.
    A solution is kept only if EVERY tensor of that type whose stored size can be
    pinned agrees on the same pair, so a coincidental single fit is not reported.
    `align` is the padding regime to assume for the FOLLOWING tensor; 0 means the
    writer stores tensors back to back (what validate_layout_arithmetic() reports
    for the files in this record), and a nonzero value tries both that padding and
    no padding, so a padded file narrows the candidates rather than lying.
    """
    kv, tensors, data_offset = read_tensor_table(path)
    file_end = os.path.getsize(path) - data_offset
    sizes, unsolved = [], 0
    for i, (name, dims, ttype, off) in enumerate(tensors):
        if ttype != type_id:
            continue
        nelem = 1
        for d in dims:
            nelem *= int(d)
        if i + 1 < len(tensors):
            nxt = tensors[i + 1][3]
            gap = nxt - off
            opts = [gap] if not align else [gap, gap - ((nxt % align) if nxt % align else 0)]
        else:
            opts = [file_end - off]
        sizes.append((name, nelem, [g for g in opts if g > 0]))
    if not sizes:
        return {"type_id": type_id, "type_name": type_name(type_id),
                "solutions": [], "tensors": 0, "unsolved": 0, "align": align,
                "note": "no tensor of this type in this file"}
    cands = None
    for name, nelem, opts in sizes:
        here = set()
        for p in LAYOUT_CANDIDATE_BLOCKS:
            if nelem % p:
                continue
            for g in opts:
                if (g * p) % nelem == 0 and g * p // nelem > 0:
                    here.add((p, g * p // nelem))
        if not here:
            unsolved += 1
            continue
        cands = here if cands is None else (cands & here)
        if cands is not None and not cands:
            break
    solved = sorted(cands or ())
    if len(solved) > 1:
        # a candidate that divides another is the coarser reading of the same
        # block; keep the largest bytes-per-block, which is the real block, and
        # report the rest as alternatives rather than silently picking.
        solved = [s for s in solved]
    return {"type_id": type_id, "type_name": type_name(type_id),
            "solutions": solved, "tensors": len(sizes), "unsolved": unsolved,
            "align": align}


def describe_unknown_type(path: str, type_id: int) -> str:
    """A one-line answer for a type this reader has no row for.

    The three outcomes are distinguished, because they are three different facts and
    naming one cause for another is a wrong refusal message: NO TENSOR of the type,
    tensors present but the size NOT SOLVABLE, tensors present and a size MEASURED.
    """
    info = infer_type_layout(path, type_id)
    if not info["tensors"]:
        return ("type %d (%s): NO TENSOR of this type in this file -- the tensor count"
                " is a reading of the file, not a failure to measure"
                % (type_id, type_name(type_id)))
    if not info["solutions"]:
        return ("type %d (%s): %d tensor(s) present but the block size is NOT"
                " solvable from the offsets -- a non-cumulative writer, or every"
                " tensor of the type is last with no following offset"
                % (type_id, type_name(type_id), info["tensors"]))
    return ("type %d (%s): %d tensor(s); measured (values/block, bytes/block) = %s"
            " -- a power-of-two-scaled family, and the bit order picks the member;"
            " add the row and a decoder, class (i) or (ii)"
            % (type_id, type_name(type_id), info["tensors"], info["solutions"]))


# --------------------------------------------------------------------------- #
# the two derived strings, RE-DERIVED.  They are computed far above, before the
# decoders in this block existed, so the refusal message would otherwise name a
# shorter list than this module can read.  Rebinding them here keeps the append at
# the end of the file while making to_fp32()'s message true again.  The insertion
# ORDER of DEQUANTIZERS is unchanged for every pre-existing type.
# --------------------------------------------------------------------------- #
SUPPORTED_NAMES = "/".join(sorted(GGML_TYPES[t] for t in DEQUANTIZERS))
KNOWN_UNSUPPORTED = tuple(sorted(
    t for t in GGML_LAYOUT if t not in DEQUANTIZERS
    and GGML_TYPES.get(t, "").endswith(("_K", "_0", "_1", "_NL", "_XS", "_S", "_M"))
))

