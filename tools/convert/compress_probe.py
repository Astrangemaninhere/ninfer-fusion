#!/usr/bin/env python3
"""compress_probe.py -- measure the *lossless* compression headroom of `.ninfer` artifacts.

Read-only probe for TODO section 10.1 ("ninfer 格式进一步压缩 -- 只做无损/保精度").
For every stored tensor it answers three questions and then rolls them up:

1. Row elimination -- bytes sitting in all-zero / all-NaN / constant rows
   (vocab padding rows, zeroed padding, degenerate rows).  Removable losslessly.
2. Scale-plane redundancy -- the NVFP4 per-16 E4M3 scale plane and the FP8 BF16
   row-scale plane: distinct words, 0-order entropy, run-length behaviour.  A
   lossless repack (RLE or a static entropy table) reclaims the gap to the
   current fixed-width storage.
3. Code-plane entropy -- NVFP4 half-byte codes / FP8 E4M3 codes: 0-order entropy
   vs the current fixed 4 / 8 bits per element.  This gap is the theoretical
   upper bound for a lossless entropy coder.

Geometry comes from the authoritative `tools.artifact.layouts` API; nothing is
re-parsed by hand.  Row blocks are streamed out of the mmap so a 20 GB artifact
can be probed on a 32 GB host.  `--sample-layers N` trades exactness for speed on
large artifacts; the scan fraction is reported and recorded in the JSON.

Usage:
    python -m tools.convert.compress_probe --artifact PATH [--json out.json]
                                          [--sample-layers N] [--sample-rows 64]
                                          [--small-mb 16] [--block-mb 32]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # allow `python tools/convert/compress_probe.py`
    sys.path.insert(0, str(_REPO_ROOT))

from tools.artifact.container import (  # noqa: E402
    Artifact,
    ResourceObject,
    TensorObject,
)
from tools.artifact.layouts import (  # noqa: E402
    BLOCKSCALE_K16_M128X4_V1,
    CONTIGUOUS_LE_V1,
    ROW_SCALE_V1,
    ROW_SPLIT_K128_V1,
    block_scale_geometry,
    get_layout,
    row_scale_geometry,
    row_split_geometry,
)
from tools.artifact.numeric import (  # noqa: E402
    DirectFormat,
    Fp8RowFormat,
    Nvfp4Format,
    QuantFormat,
    get_format,
)

BYTE_HIST = 256
WORD_HIST = 1 << 16
_NAN_FP8 = np.zeros(BYTE_HIST, dtype=bool)
_NAN_FP8[0x7F] = True
_NAN_FP8[0xFF] = True


# --------------------------------------------------------------------------- #
# entropy helpers
# --------------------------------------------------------------------------- #
def entropy_bits(counts: np.ndarray) -> float:
    """0-order Shannon entropy (bits/symbol) of a histogram."""
    total = int(counts.sum())
    if total <= 0:
        return 0.0
    nz = counts[counts > 0].astype(np.float64)
    p = nz / total
    return float(-(p * np.log2(p)).sum())


def _bincount_into(acc: np.ndarray, values: np.ndarray, size: int) -> None:
    flat = values.reshape(-1)
    step = 1 << 23
    for begin in range(0, flat.size, step):
        chunk = flat[begin : begin + step]
        if chunk.dtype == np.uint8:
            acc += np.bincount(chunk, minlength=size)
        else:
            acc += np.bincount(chunk.astype(np.int64), minlength=size)


def _count_runs(values: np.ndarray, carry: int | None) -> tuple[int, int | None]:
    flat = values.reshape(-1)
    if flat.size == 0:
        return 0, carry
    if flat.size == 1:
        return 1, int(flat[0])
    runs = int(np.count_nonzero(flat[1:] != flat[:-1])) + 1
    if carry is not None and int(flat[0]) == carry:
        runs -= 1
    return runs, int(flat[-1])


# --------------------------------------------------------------------------- #
# plane accumulator
# --------------------------------------------------------------------------- #
@dataclass
class PlaneAccum:
    """Histogram + run-length accounting for one logical plane."""

    label: str
    word_bits: int
    hist_size: int
    plane_bytes: int = 0      # bytes of the plane inside the scanned rows (incl. zero rows)
    kept_bytes: int = 0       # bytes fed to the histogram (zero rows excluded)
    words: int = 0            # symbols fed to the histogram
    runs: int = 0
    hist: np.ndarray = field(default=None)  # type: ignore[assignment]
    carry: int | None = None

    def __post_init__(self) -> None:
        if self.hist is None:
            self.hist = np.zeros(self.hist_size, dtype=np.int64)

    def add(self, values: np.ndarray, plane_bytes: int, kept_bytes: int) -> None:
        self.plane_bytes += plane_bytes
        self.kept_bytes += kept_bytes
        self.words += int(values.size)
        _bincount_into(self.hist, values, self.hist_size)
        runs, self.carry = _count_runs(values, self.carry)
        self.runs += runs

    def merge(self, other: "PlaneAccum") -> None:
        self.hist += other.hist
        self.plane_bytes += other.plane_bytes
        self.kept_bytes += other.kept_bytes
        self.words += other.words
        self.runs += other.runs
        self.carry = None

    def finalize(self) -> dict:
        distinct = int(np.count_nonzero(self.hist))
        top1 = int(self.hist.max()) if self.words else 0
        bits = entropy_bits(self.hist)
        ent_bytes = bits * self.words / 8.0
        rle_bytes = self.runs * (1 + self.word_bits // 8)
        return {
            "label": self.label,
            "word_bits": self.word_bits,
            "plane_bytes": self.plane_bytes,
            "kept_bytes": self.kept_bytes,
            "words": self.words,
            "distinct": distinct,
            "top1_share": round(top1 / self.words, 6) if self.words else 0.0,
            "entropy_bits_per_word": round(bits, 6),
            "entropy_bytes": round(ent_bytes, 1),
            "entropy_savings_bytes": round(self.kept_bytes - ent_bytes, 1),
            "entropy_savings_frac": round(1.0 - ent_bytes / self.kept_bytes, 6)
            if self.kept_bytes
            else 0.0,
            "runs": self.runs,
            "rle_bytes": rle_bytes,
            "rle_savings_bytes": round(self.kept_bytes - rle_bytes, 1),
            "rle_savings_frac": round(1.0 - rle_bytes / self.kept_bytes, 6)
            if self.kept_bytes
            else 0.0,
        }


# --------------------------------------------------------------------------- #
# per-tensor probe
# --------------------------------------------------------------------------- #
class TensorProbe:
    def __init__(self, artifact: Artifact, obj: TensorObject, block_mb: int):
        self.artifact = artifact
        self.obj = obj
        self.block_mb = block_mb
        self.format = get_format(obj.format)
        self.layout = get_layout(obj.layout)

    def _block_rows(self, row_bytes: int, n: int) -> int:
        if row_bytes <= 0:
            return n
        return max(1, (self.block_mb * 1024 * 1024) // row_bytes)

    def run(self, rows: Sequence[int] | None) -> dict:
        if self.layout is CONTIGUOUS_LE_V1:
            return self._contiguous(rows)
        if self.layout is ROW_SCALE_V1:
            return self._row_scale(rows)
        if self.layout is BLOCKSCALE_K16_M128X4_V1:
            return self._blockscale(rows)
        if self.layout is ROW_SPLIT_K128_V1:
            return self._row_split(rows)
        return {"name": self.obj.name, "unsupported": self.layout.name}

    def _spans(
        self, n: int, row_bytes: int, rows: Sequence[int] | None, multiple: int = 1
    ):
        if rows is None:
            step = self._block_rows(row_bytes, n)
            if multiple > 1:
                step = max(multiple, step // multiple * multiple)
            return [(b, min(step, n - b)) for b in range(0, n, step)]
        return [(int(r), 1) for r in rows if 0 <= int(r) < n]

    # -- contiguous-le-v1 (BF16 / FP32 / I32) ------------------------------ #
    def _contiguous(self, rows: Sequence[int] | None) -> dict:
        spec = self.format
        assert isinstance(spec, DirectFormat)
        obj = self.obj
        blob = self.artifact.payload(obj)
        if len(obj.shape) < 2:
            n, row_bytes = 1, obj.bytes
        else:
            n = obj.shape[0]
            row_bytes = obj.bytes // n
        wbytes = spec.word_bytes
        hist_size = WORD_HIST if wbytes == 2 else BYTE_HIST
        code = PlaneAccum("codes", wbytes * 8, hist_size)
        zero_rows = const_rows = nan_rows = zero_code_rows = 0
        scanned = 0
        for begin, count in self._spans(n, row_bytes, rows):
            chunk = blob[begin * row_bytes : (begin + count) * row_bytes]
            mat = np.frombuffer(chunk, dtype=np.uint8).reshape(count, row_bytes)
            scanned += count
            nz = np.count_nonzero(mat, axis=1)
            zero_mask = nz == 0
            zero_rows += int(zero_mask.sum())
            zero_code_rows += int(zero_mask.sum())
            if row_bytes > 1:
                cmask = mat.max(axis=1) == mat.min(axis=1)
                const_rows += int(cmask.sum()) - int(zero_mask.sum())
            if wbytes == 2:
                words = np.frombuffer(chunk, dtype="<u2").reshape(count, row_bytes // 2)
                nan = (((words >> 7) & 0xFF) == 0xFF) & ((words & 0x7F) != 0)
                nan_rows += int(nan.any(axis=1).sum())
            elif wbytes == 4:
                words = np.frombuffer(chunk, dtype="<u4").reshape(count, row_bytes // 4)
                nan = (((words >> 23) & 0xFF) == 0xFF) & ((words & 0x7FFFFF) != 0)
                nan_rows += int(nan.any(axis=1).sum())
            keep = ~zero_mask
            if keep.any():
                sel = mat[keep]
                code.add(
                    sel.view(np.uint16).reshape(-1) if wbytes == 2 else sel.reshape(-1),
                    int(keep.sum()) * row_bytes,
                    int(keep.sum()) * row_bytes,
                )
        return self._result(
            n, row_bytes, row_bytes, scanned, zero_rows, zero_code_rows, nan_rows,
            const_rows, code, None, None,
        )

    # -- row-scale-v1 (FP8 E4M3FN + BF16 row multiplier) ------------------- #
    def _row_scale(self, rows: Sequence[int] | None) -> dict:
        spec = self.format
        assert isinstance(spec, Fp8RowFormat)
        geo = row_scale_geometry(spec, self.obj.shape)
        blob = self.artifact.payload(self.obj)
        code = PlaneAccum("codes", 8, BYTE_HIST)
        scale = PlaneAccum("scales", 16, WORD_HIST)
        codes_mv = blob[: geo.code_plane_bytes]
        scale_mv = blob[geo.scale_plane_offset : geo.scale_plane_offset + geo.scale_plane_bytes]
        n, k = geo.n, geo.k
        zero_rows = const_rows = nan_rows = nan_scales = zero_code_rows = 0
        scanned = 0
        for begin, count in self._spans(n, k + 2, rows):
            cblock = codes_mv[begin * k : (begin + count) * k]
            sblock = scale_mv[begin * 2 : (begin + count) * 2]
            codes = np.frombuffer(cblock, dtype=np.uint8).reshape(count, k)
            scales = np.frombuffer(sblock, dtype="<u2").reshape(count)
            scanned += count
            mag = codes & 0x7F
            zero_code_mask = np.count_nonzero(mag, axis=1) == 0
            zero_mask = zero_code_mask & (scales == 0)
            zero_code_rows += int(zero_code_mask.sum())
            zero_rows += int(zero_mask.sum())
            flat = codes.max(axis=1) == codes.min(axis=1)
            const_rows += int(flat.sum()) - int(zero_code_mask.sum())
            nan_rows += int(_NAN_FP8[codes].any(axis=1).sum())
            nan_scales += int(((scales & 0x7F80) == 0x7F80).sum())
            code_keep = ~zero_code_mask
            if code_keep.any():
                sel = codes[code_keep]
                code.add(sel.reshape(-1), int(code_keep.sum()) * k, int(code_keep.sum()) * k)
            scale_keep = ~zero_mask
            if scale_keep.any():
                scale.add(
                    scales[scale_keep], int(scale_keep.sum()) * 2, int(scale_keep.sum()) * 2
                )
        res = self._result(
            n, k + 2, k, scanned, zero_rows, zero_code_rows, nan_rows, const_rows,
            code, scale, None,
        )
        res["row_stats"]["nan_scale_words"] = nan_scales
        return res

    # -- blockscale-k16-m128x4-v1 (NVFP4) ---------------------------------- #
    def _blockscale(self, rows: Sequence[int] | None) -> dict:
        spec = self.format
        assert isinstance(spec, Nvfp4Format)
        geo = block_scale_geometry(spec, self.obj.shape)
        blob = self.artifact.payload(self.obj)
        code = PlaneAccum("codes", 8, BYTE_HIST)
        nib = PlaneAccum("nibbles", 4, 16)
        scale = PlaneAccum("scales", 8, BYTE_HIST)
        codes_mv = blob[: geo.code_plane_bytes]
        scale_mv = blob[geo.scale_plane_offset : geo.scale_plane_offset + geo.scale_plane_bytes]
        n, k = geo.n, geo.k
        half_k, gpr, k_tiles = k // 2, geo.groups_per_row, geo.k_tiles
        zero_rows = const_rows = zero_code_rows = 0
        scanned = 0
        for begin, count in self._spans(n, half_k + gpr, rows, multiple=128):
            cblock = codes_mv[begin * half_k : (begin + count) * half_k]
            codes = np.frombuffer(cblock, dtype=np.uint8).reshape(count, half_k)
            scanned += count
            zero_code_mask = np.count_nonzero(codes, axis=1) == 0
            nat = self._natural_scales(scale_mv, begin, count, gpr, k_tiles)
            zero_mask = zero_code_mask & (np.count_nonzero(nat, axis=1) == 0)
            zero_code_rows += int(zero_code_mask.sum())
            zero_rows += int(zero_mask.sum())
            cmask = codes.max(axis=1) == codes.min(axis=1)
            const_rows += int(cmask.sum()) - int(zero_code_mask.sum())
            keep = ~zero_code_mask
            if keep.any():
                sel = codes[keep]
                kb = int(keep.sum())
                code.add(sel.reshape(-1), kb * half_k, kb * half_k)
                lo = (sel & 0x0F).reshape(-1)
                hi = (sel >> 4).reshape(-1)
                nib.add(lo, 0, 0)
                nib.add(hi, 0, 0)
            scale_keep = ~zero_mask
            if scale_keep.any():
                kb = int(scale_keep.sum())
                scale.add(nat[scale_keep].reshape(-1), kb * gpr, kb * gpr)
        return self._result(
            n, half_k + gpr, half_k, scanned, zero_rows, zero_code_rows, 0,
            const_rows, code, scale, nib,
        )

    @staticmethod
    def _natural_scales(
        scale_mv: memoryview, begin: int, count: int, gpr: int, k_tiles: int
    ) -> np.ndarray:
        """Unswizzle NVFP4 scales back to natural [count, groups_per_row]."""
        if begin % 128 == 0 and count % 128 == 0:
            blocks = count // 128
            sview = np.frombuffer(
                scale_mv[(begin // 128) * 128 * gpr : (begin // 128 + blocks) * 128 * gpr],
                dtype=np.uint8,
            ).reshape(blocks, k_tiles, 32, 4, 4)
            return sview.transpose(0, 3, 2, 1, 4).reshape(count, gpr)
        out = np.empty((count, gpr), dtype=np.uint8)
        for i in range(count):
            r = begin + i
            blk, off = divmod(r, 128)
            row_view = np.frombuffer(
                scale_mv[blk * 128 * gpr : (blk + 1) * 128 * gpr], dtype=np.uint8
            ).reshape(k_tiles, 32, 4, 4)
            out[i] = row_view[:, off // 4, off % 4, :].reshape(-1)
        return out

    # -- row-split-k128-v1 (Q4/Q5/Q6/W8 grouped codes) --------------------- #
    def _row_split(self, rows: Sequence[int] | None) -> dict:
        spec = self.format
        assert isinstance(spec, QuantFormat)
        geo = row_split_geometry(spec, self.obj.shape)
        blob = self.artifact.payload(self.obj)
        code = PlaneAccum("codes", 8, BYTE_HIST)
        scale = PlaneAccum("scales", 16, WORD_HIST)
        base_mv = blob[geo.base_offset : geo.base_offset + geo.base_bytes]
        high_mv = blob[geo.high_offset : geo.high_offset + geo.high_bytes]
        scale_mv = blob[geo.scale_offset : geo.scale_offset + geo.scale_bytes]
        n = geo.n
        row_bytes = geo.base_row_bytes + geo.high_row_bytes + geo.scale_row_bytes
        code_row_bytes = geo.base_row_bytes + geo.high_row_bytes
        zero_rows = const_rows = zero_code_rows = 0
        scanned = 0
        for begin, count in self._spans(n, row_bytes, rows):
            b = np.frombuffer(
                base_mv[begin * geo.base_row_bytes : (begin + count) * geo.base_row_bytes],
                dtype=np.uint8,
            ).reshape(count, geo.base_row_bytes)
            s = np.frombuffer(
                scale_mv[begin * geo.scale_row_bytes : (begin + count) * geo.scale_row_bytes],
                dtype="<u2",
            ).reshape(count, geo.groups_per_row)
            if geo.high_row_bytes:
                h = np.frombuffer(
                    high_mv[begin * geo.high_row_bytes : (begin + count) * geo.high_row_bytes],
                    dtype=np.uint8,
                ).reshape(count, geo.high_row_bytes)
                zero_code_mask = (np.count_nonzero(b, axis=1) == 0) & (
                    np.count_nonzero(h, axis=1) == 0
                )
                zero_mask = zero_code_mask & (np.count_nonzero(s, axis=1) == 0)
                cmask = (
                    (b.max(axis=1) == b.min(axis=1))
                    & (h.max(axis=1) == h.min(axis=1))
                    & (s.max(axis=1) == s.min(axis=1))
                )
            else:
                h = None
                zero_code_mask = np.count_nonzero(b, axis=1) == 0
                zero_mask = zero_code_mask & (np.count_nonzero(s, axis=1) == 0)
                cmask = (b.max(axis=1) == b.min(axis=1)) & (s.max(axis=1) == s.min(axis=1))
            scanned += count
            zero_code_rows += int(zero_code_mask.sum())
            zero_rows += int(zero_mask.sum())
            const_rows += int(cmask.sum()) - int(zero_code_mask.sum())
            code_keep = ~zero_code_mask
            if code_keep.any():
                kb = int(code_keep.sum())
                code.add(b[code_keep].reshape(-1), kb * geo.base_row_bytes, kb * geo.base_row_bytes)
                if h is not None:
                    code.add(
                        h[code_keep].reshape(-1),
                        kb * geo.high_row_bytes,
                        kb * geo.high_row_bytes,
                    )
            scale_keep = ~zero_mask
            if scale_keep.any():
                kb = int(scale_keep.sum())
                scale.add(s[scale_keep].reshape(-1), kb * geo.scale_row_bytes, kb * geo.scale_row_bytes)
        return self._result(
            n, row_bytes, code_row_bytes, scanned, zero_rows, zero_code_rows, 0,
            const_rows, code, scale, None,
        )

    # -- shared result assembly -------------------------------------------- #
    def _result(
        self,
        n: int,
        row_bytes: int,
        code_row_bytes: int,
        scanned: int,
        zero_rows: int,
        zero_code_rows: int,
        nan_rows: int,
        const_rows: int,
        code: PlaneAccum,
        scale: PlaneAccum | None,
        nib: PlaneAccum | None,
    ) -> dict:
        return {
            "name": self.obj.name,
            "shape": list(self.obj.shape),
            "format": self.obj.format,
            "layout": self.obj.layout,
            "bytes": self.obj.bytes,
            "rows": {"scanned": scanned, "total": n},
            "row_bytes": row_bytes,
            "code_row_bytes": code_row_bytes,
            "row_stats": {
                "zero_rows": zero_rows,
                "zero_code_rows": zero_code_rows,
                "nan_rows": nan_rows,
                "constant_rows_excl_zero": const_rows,
                # a fully zero row drops every plane; a zero-code row drops only
                # the code plane (a nonzero constant scale must be retained)
                "zero_row_bytes": zero_rows * row_bytes,
                "zero_code_row_bytes": zero_code_rows * code_row_bytes,
                "removable_zero_bytes": zero_rows * row_bytes
                + (zero_code_rows - zero_rows) * code_row_bytes,
                "nan_row_bytes": nan_rows * row_bytes,
                "constant_row_bytes": const_rows * row_bytes,
            },
            "code": code,
            "scale": scale,
            "nibbles": nib,
        }


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _select_rows(n: int, sample_rows: int) -> list[int]:
    """Anchored sample: head, tail (padding lives there) + an even sweep."""
    if sample_rows >= n:
        return list(range(n))
    head = min(32, n)
    tail = min(256, n - head)
    span = n - head - tail
    mid = np.linspace(0, span - 1, max(1, sample_rows - head - tail), dtype=np.int64)
    rows = set(range(head)) | set(range(n - tail, n)) | {int(x) for x in mid}
    return sorted(rows)


def _pick_tensors(
    tensors: list[TensorObject], small_bytes: int, sample_layers: int | None
) -> tuple[list[TensorObject], list[TensorObject]]:
    small = [o for o in tensors if o.bytes <= small_bytes]
    large = [o for o in tensors if o.bytes > small_bytes]
    if sample_layers is None or len(large) <= sample_layers:
        return small + large, []
    groups: dict[tuple[str, str], list[TensorObject]] = {}
    for obj in large:
        groups.setdefault((obj.format, obj.layout), []).append(obj)
    picked: list[TensorObject] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda o: o.name)
        if len(members) <= sample_layers:
            picked.extend(members)
            continue
        idx = np.linspace(0, len(members) - 1, sample_layers, dtype=np.int64)
        picked.extend(members[int(i)] for i in sorted(set(int(i) for i in idx)))
    chosen = {id(o) for o in picked}
    sampled = [o for o in large if id(o) not in chosen]
    return small + picked, sampled


def _merge_plane(acc: PlaneAccum | None, other: PlaneAccum) -> PlaneAccum:
    """Fold `other` into a group accumulator of matching word geometry."""
    if acc is None:
        acc = PlaneAccum(other.label, other.word_bits, other.hist_size)
    elif acc.hist_size != other.hist_size or acc.word_bits != other.word_bits:
        raise ValueError(
            f"plane {other.label} geometry mismatch: "
            f"{acc.hist_size}/{acc.word_bits} vs {other.hist_size}/{other.word_bits}"
        )
    acc.merge(other)
    return acc


def _codec_check(
    artifact: Artifact, groups: dict, codec_mb: int, block_mb: int
) -> None:
    """zlib-compress a bounded sample of each group's planes.

    The 0-order entropy is a theoretical bound; this records what an off-the-shelf
    lossless coder actually reaches on the same bytes, which is what a layout
    extension would realistically ship.
    """
    import zlib

    limit = codec_mb * 1024 * 1024
    for key, ga in groups.items():
        rep = ga.get("largest")
        if rep is None:
            continue
        try:
            layout = get_layout(rep.layout)
            fmt = get_format(rep.format)
        except ValueError:
            continue
        blob = artifact.payload(rep)
        planes: dict[str, bytes] = {}
        if layout is ROW_SCALE_V1:
            geo = row_scale_geometry(fmt, rep.shape)
            planes["code"] = bytes(blob[: min(limit, geo.code_plane_bytes)])
            planes["scale"] = bytes(
                blob[geo.scale_plane_offset : geo.scale_plane_offset + geo.scale_plane_bytes][
                    :limit
                ]
            )
        elif layout is BLOCKSCALE_K16_M128X4_V1:
            geo = block_scale_geometry(fmt, rep.shape)
            planes["code"] = bytes(blob[: min(limit, geo.code_plane_bytes)])
            planes["scale"] = bytes(
                blob[geo.scale_plane_offset : geo.scale_plane_offset + geo.scale_plane_bytes][
                    :limit
                ]
            )
        elif layout is ROW_SPLIT_K128_V1:
            geo = row_split_geometry(fmt, rep.shape)
            planes["code"] = bytes(blob[: min(limit, geo.base_bytes)])
            planes["scale"] = bytes(
                blob[geo.scale_offset : geo.scale_offset + geo.scale_bytes][:limit]
            )
        elif layout is CONTIGUOUS_LE_V1:
            planes["code"] = bytes(blob[: min(limit, rep.bytes)])
        out: dict[str, dict] = {}
        for name, data in planes.items():
            if not data:
                continue
            packed = zlib.compress(data, 6)
            out[name] = {
                "sample_bytes": len(data),
                "zlib_bytes": len(packed),
                "zlib_saved_frac": round(1.0 - len(packed) / len(data), 6),
            }
        if out:
            out["tensor"] = rep.name
            ga["codec"] = out


def _container_overhead(artifact: Artifact) -> dict:
    gap = 0
    cursor = 0
    for obj in artifact.objects:
        gap += obj.offset - cursor
        cursor = obj.offset + obj.bytes
    pad = 0
    for obj in artifact.objects:
        if not isinstance(obj, TensorObject):
            continue
        try:
            layout = get_layout(obj.layout)
            fmt = get_format(obj.format)
        except ValueError:
            continue
        if layout is CONTIGUOUS_LE_V1:
            continue
        if layout is ROW_SCALE_V1:
            geo = row_scale_geometry(fmt, obj.shape)
            pad += obj.bytes - geo.scale_plane_offset - geo.scale_plane_bytes
        elif layout is BLOCKSCALE_K16_M128X4_V1:
            geo = block_scale_geometry(fmt, obj.shape)
            pad += geo.scale_plane_offset - geo.code_plane_bytes
            pad += (
                geo.weight_divisor_offset
                + 4
                - geo.scale_plane_offset
                - geo.scale_plane_bytes
            )
        elif layout is ROW_SPLIT_K128_V1:
            geo = row_split_geometry(fmt, obj.shape)
            pad += geo.high_offset - geo.base_bytes
            pad += geo.scale_offset - geo.high_offset - geo.high_bytes
    return {"inter_object_gap_bytes": gap, "intra_plane_pad_bytes": pad}


# --------------------------------------------------------------------------- #
# artifact walk
# --------------------------------------------------------------------------- #
def probe_artifact(
    path: Path,
    sample_layers: int | None = None,
    sample_rows: int = 64,
    small_mb: int = 16,
    block_mb: int = 32,
    codec_mb: int = 0,
) -> dict:
    started = time.time()
    with Artifact.open(path) as artifact:
        tensors = [o for o in artifact.objects if isinstance(o, TensorObject)]
        resources = [o for o in artifact.objects if isinstance(o, ResourceObject)]
        full, sampled = _pick_tensors(tensors, small_mb * 1024 * 1024, sample_layers)
        full_ids = {id(o) for o in full}

        groups: dict[tuple[str, str], dict] = {}
        per_tensor: list[dict] = []
        total_scanned_bytes = 0
        total_tensor_bytes = 0

        for obj in artifact.objects:
            if not isinstance(obj, TensorObject):
                continue
            total_tensor_bytes += obj.bytes
            key = (obj.format, obj.layout)
            ga = groups.setdefault(
                key,
                {
                    "format": obj.format,
                    "layout": obj.layout,
                    "tensors": 0,
                    "tensor_bytes": 0,
                    "zero_row_bytes": 0,
                    "nan_row_bytes": 0,
                    "constant_row_bytes": 0,
                    "scanned_rows": 0,
                    "total_rows": 0,
                    "code": None,
                    "scale": None,
                    "nibbles": None,
                },
            )
            ga["tensors"] += 1
            ga["tensor_bytes"] += obj.bytes
            if ga.get("largest") is None or obj.bytes > ga["largest"].bytes:
                ga["largest"] = obj
            n = obj.shape[0] if obj.shape else 1
            ga["total_rows"] += n
            is_full = id(obj) in full_ids
            probe = TensorProbe(artifact, obj, block_mb)
            try:
                rows = None if is_full else _select_rows(n, sample_rows)
                res = probe.run(rows)
            except Exception as exc:
                res = {
                    "name": obj.name,
                    "shape": list(obj.shape),
                    "format": obj.format,
                    "layout": obj.layout,
                    "bytes": obj.bytes,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            res["fully_scanned"] = is_full
            if "error" not in res:
                scanned = res["rows"]["scanned"]
                res["scan_fraction"] = round(scanned / n, 6) if n else 1.0
                res["scanned_bytes"] = int(obj.bytes * res["scan_fraction"])
                total_scanned_bytes += res["scanned_bytes"]
                ga["scanned_rows"] += scanned
                rs = res["row_stats"]
                ga["zero_row_bytes"] += rs["removable_zero_bytes"]
                ga["nan_row_bytes"] += rs["nan_row_bytes"]
                ga["constant_row_bytes"] += rs["constant_row_bytes"]
                ga["code"] = _merge_plane(ga["code"], res["code"])
                if res["scale"] is not None:
                    ga["scale"] = _merge_plane(ga["scale"], res["scale"])
                if res["nibbles"] is not None:
                    ga["nibbles"] = _merge_plane(ga["nibbles"], res["nibbles"])
            per_tensor.append(_tensor_json(res))

        group_rows = []
        if codec_mb > 0:
            _codec_check(artifact, groups, codec_mb, block_mb)
        for key, ga in sorted(groups.items()):
            if ga["code"] is None:
                continue
            code = ga["code"].finalize()
            scale = ga["scale"].finalize() if ga["scale"] is not None else None
            nib = ga["nibbles"].finalize() if ga["nibbles"] is not None else None
            code_save = code["entropy_savings_bytes"]
            nib_save = None
            if nib is not None:
                # nibble symbols == 2 per packed byte; kept_bytes is the code plane
                nib_bytes = nib["entropy_bits_per_word"] * nib["words"] / 8.0
                nib_save = round(code["kept_bytes"] - nib_bytes, 1)
                code_save = max(code_save, nib_save)
            scale_save = scale["entropy_savings_bytes"] if scale else 0.0
            rec = ga["zero_row_bytes"] + code_save + scale_save
            group_rows.append(
                {
                    "format": ga["format"],
                    "layout": ga["layout"],
                    "tensors": ga["tensors"],
                    "tensor_bytes": ga["tensor_bytes"],
                    "scanned_rows": ga["scanned_rows"],
                    "total_rows": ga["total_rows"],
                    "zero_row_bytes": ga["zero_row_bytes"],
                    "nan_row_bytes": ga["nan_row_bytes"],
                    "constant_row_bytes": ga["constant_row_bytes"],
                    "code": code,
                    "scale": scale,
                    "nibble": nib,
                    "code_savings_bytes": code_save,
                    "scale_savings_bytes": scale_save,
                    "recoverable_bytes": round(rec, 1),
                    "codec": ga.get("codec"),
                    "recoverable_frac_of_group": round(rec / ga["tensor_bytes"], 6)
                    if ga["tensor_bytes"]
                    else 0.0,
                }
            )

        file_bytes = artifact.file_bytes
        total_rec = sum(g["recoverable_bytes"] for g in group_rows)
        overhead = _container_overhead(artifact)
        practical = 0.0
        for g in group_rows:
            codec = g["codec"]
            if not codec:
                continue
            code_frac = codec.get("code", {}).get("zlib_saved_frac", 0.0)
            scale_frac = codec.get("scale", {}).get("zlib_saved_frac", 0.0)
            practical += g["zero_row_bytes"]
            practical += g["code"]["kept_bytes"] * code_frac
            if g["scale"]:
                practical += g["scale"]["kept_bytes"] * scale_frac
        summary = {
            "path": str(path),
            "model_id": artifact.identity.model_id,
            "weights_id": artifact.identity.weights_id,
            "file_bytes": file_bytes,
            "payload_offset": artifact.payload_offset,
            "tensor_bytes": total_tensor_bytes,
            "resource_bytes": sum(o.bytes for o in resources),
            "container_overhead": overhead,
            "tensors": len(tensors),
            "fully_scanned_tensors": len(full),
            "sampled_tensors": len(sampled),
            "scanned_bytes": total_scanned_bytes,
            "scan_fraction_of_tensor_bytes": round(
                total_scanned_bytes / total_tensor_bytes, 6
            )
            if total_tensor_bytes
            else 1.0,
            "recoverable_bytes": round(total_rec, 1),
            "recoverable_pct_of_file": round(100.0 * total_rec / file_bytes, 4),
            "practical_zlib_bytes": round(practical, 1),
            "practical_zlib_pct_of_file": round(100.0 * practical / file_bytes, 4),
            "elapsed_s": round(time.time() - started, 1),
        }
        return {"summary": summary, "groups": group_rows, "tensors": per_tensor}


def _tensor_json(res: dict) -> dict:
    out = {
        "name": res["name"],
        "shape": res.get("shape"),
        "format": res.get("format"),
        "layout": res.get("layout"),
        "bytes": res.get("bytes"),
        "fully_scanned": res.get("fully_scanned"),
        "scan_fraction": res.get("scan_fraction"),
    }
    if "error" in res:
        out["error"] = res["error"]
        return out
    out["row_stats"] = res["row_stats"]
    out["code"] = res["code"].finalize()
    out["scale"] = res["scale"].finalize() if res["scale"] is not None else None
    out["nibble"] = res["nibbles"].finalize() if res["nibbles"] is not None else None
    rec = (
        res["row_stats"]["removable_zero_bytes"]
        + out["code"]["entropy_savings_bytes"]
        + (out["scale"]["entropy_savings_bytes"] if out["scale"] else 0.0)
    )
    if out["nibble"] is not None:
        nib_bytes = out["nibble"]["entropy_bits_per_word"] * out["nibble"]["words"] / 8.0
        rec = max(
            rec,
            res["row_stats"]["removable_zero_bytes"]
            + (out["code"]["kept_bytes"] - nib_bytes)
            + (out["scale"]["entropy_savings_bytes"] if out["scale"] else 0.0),
        )
    out["recoverable_bytes"] = round(rec, 1)
    out["recoverable_frac"] = round(rec / res["bytes"], 6) if res["bytes"] else 0.0
    return out


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    x = float(x)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024 or unit == "GB":
            return f"{x:.2f}{unit}"
        x /= 1024.0
    return f"{x:.2f}GB"


def render(report: dict, top: int = 12) -> str:
    s = report["summary"]
    L: list[str] = []
    L.append(f"artifact : {s['path']}")
    L.append(
        f"identity : {s['model_id']} / {s['weights_id']}"
        f"    file {_fmt(s['file_bytes'])}    tensors {s['tensors']}"
    )
    L.append(
        f"scan     : {s['fully_scanned_tensors']} full + {s['sampled_tensors']} sampled"
        f" = {100.0*s['scan_fraction_of_tensor_bytes']:.1f}% of tensor bytes"
        f"    ({s['elapsed_s']}s)"
    )
    L.append("")
    hdr = (
        f"{'format/layout':<44}{'tens':>5}{'bytes':>11}"
        f"{'zero%':>8}{'scale%':>8}{'code%':>8}{'recov%':>9}{'recov':>10}"
    )
    L.append(hdr)
    L.append("-" * len(hdr))
    for g in sorted(report["groups"], key=lambda x: -x["tensor_bytes"]):
        tb = g["tensor_bytes"] or 1
        L.append(
            f"{(g['format'] + '/' + g['layout']):<44}{g['tensors']:>5}"
            f"{_fmt(g['tensor_bytes']):>11}"
            f"{100.0*g['zero_row_bytes']/tb:>7.3f}%"
            f"{100.0*g['scale_savings_bytes']/tb:>7.3f}%"
            f"{100.0*g['code_savings_bytes']/tb:>7.3f}%"
            f"{100.0*g['recoverable_bytes']/tb:>8.3f}%"
            f"{_fmt(g['recoverable_bytes']):>10}"
        )
    L.append("")
    for g in sorted(report["groups"], key=lambda x: -x["tensor_bytes"]):
        c = g["code"]
        line = (
            f"[{g['format']}] codes {_fmt(c['kept_bytes'])} kept  "
            f"H={c['entropy_bits_per_word']:.4f}b/{c['word_bits']}b  "
            f"distinct={c['distinct']}"
        )
        if g["nibble"] is not None:
            line += f"  nibbleH={g['nibble']['entropy_bits_per_word']:.4f}b/4b"
        L.append(line)
        if g["scale"]:
            sc = g["scale"]
            L.append(
                f"[{g['format']}] scales {_fmt(sc['kept_bytes'])} kept  "
                f"H={sc['entropy_bits_per_word']:.4f}b/{sc['word_bits']}b  "
                f"distinct={sc['distinct']}  top1={100.0*sc['top1_share']:.1f}%  "
                f"runs={sc['runs']} (rle {_fmt(sc['rle_bytes'])}, "
                f"{100.0*sc['rle_savings_frac']:.1f}% saved)"
            )
    L.append("")
    oh = s["container_overhead"]
    L.append(
        f"container overhead: inter-object gap {_fmt(oh['inter_object_gap_bytes'])}"
        f" + intra-plane pad {_fmt(oh['intra_plane_pad_bytes'])}"
    )
    L.append(
        f"TOTAL losslessly recoverable: {_fmt(s['recoverable_bytes'])}"
        f" = {s['recoverable_pct_of_file']:.3f}% of file   (0-order entropy bound)"
    )
    if s.get("practical_zlib_bytes"):
        L.append(
            f"TOTAL reachable with zlib : {_fmt(s['practical_zlib_bytes'])}"
            f" = {s['practical_zlib_pct_of_file']:.3f}% of file   (measured zlib sample ratios)"
        )
        for g in sorted(report["groups"], key=lambda x: -x["tensor_bytes"]):
            codec = g.get("codec")
            if not codec:
                continue
            bits = " ".join(
                f"{k}={100.0*codec[k]['zlib_saved_frac']:.2f}%"
                for k in ("code", "scale")
                if k in codec
            )
            L.append(f"  zlib[{g['format']}] {bits}  (sample {codec['tensor']})")
    L.append("")
    L.append(f"top tensors by recoverable bytes (of {len(report['tensors'])}):")
    scored = [t for t in report["tensors"] if "recoverable_bytes" in t]
    for t in sorted(scored, key=lambda x: -x["recoverable_bytes"])[:top]:
        rs = t["row_stats"]
        L.append(
            f"  {_fmt(t['recoverable_bytes']):>10} ({100.0*t['recoverable_frac']:5.2f}%) "
            f"{t['format']:<26}{t['shape']} {t['name']}"
        )
        extra = (
            f"codeH={t['code']['entropy_bits_per_word']:.4f}"
            + (f" nibH={t['nibble']['entropy_bits_per_word']:.4f}" if t["nibble"] else "")
            + (
                f" scaleH={t['scale']['entropy_bits_per_word']:.4f}"
                f" scale_distinct={t['scale']['distinct']}"
                if t["scale"]
                else ""
            )
        )
        L.append(
            f"             zero={rs['zero_rows']} zero_code={rs['zero_code_rows']}"
            f" nan={rs['nan_rows']} const={rs['constant_rows_excl_zero']} {extra}"
        )
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--json", type=Path, default=None, help="write the full report as JSON")
    ap.add_argument(
        "--sample-layers",
        type=int,
        default=None,
        help="fully scan at most N large tensors per (format, layout) group",
    )
    ap.add_argument("--sample-rows", type=int, default=64, help="rows per sampled tensor")
    ap.add_argument("--small-mb", type=int, default=16, help="always fully scan below this size")
    ap.add_argument("--block-mb", type=int, default=32, help="streaming row-block size")
    ap.add_argument(
        "--codec-mb",
        type=int,
        default=0,
        help="zlib-compress this many MB of each group's largest tensor planes (0=off)",
    )
    ap.add_argument("--top", type=int, default=12, help="rows in the top-tensor table")
    args = ap.parse_args(argv)

    report = probe_artifact(
        args.artifact,
        sample_layers=args.sample_layers,
        sample_rows=args.sample_rows,
        small_mb=args.small_mb,
        block_mb=args.block_mb,
        codec_mb=args.codec_mb,
    )
    print(render(report, args.top))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
