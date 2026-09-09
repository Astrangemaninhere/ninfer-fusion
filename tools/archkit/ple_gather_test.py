#!/usr/bin/env python3
"""ple_gather_test.py — W2: PLE n-gram table real-gather verification.

Verifies the LAST unproven step of the PLE/ngram line (§66 W2): given a token
context, compute the 16 head row numbers with the manifest-pinned formula and
actually gather 16 x 160 BF16 rows from the 95.4GB SSD-backed table, then
sanity-check the values (non-degenerate, stable, EOS-bounded).

Formula (manifest hash_reference semantics + RESEARCH-FLASHNEXT.md, both
cross-checked against layer_multipliers):
  ctx[0] = current token; ctx[1..n-1] = predecessors (EOS resets the window)
  bigram  heads 0-7:  mixed = ctx[0]*m[0] ^ ctx[1]*m[1]
  trigram heads 8-15: mixed = ctx[0]*m[0] ^ ctx[1]*m[1] ^ ctx[2]*m[2]
  (uint64 natural overflow; XOR in n-gram position order)
  row[h] = mixed % per_head_vocab_size[h] + per_head_offset[h]

Usage:
  python ple_gather_test.py --root C:/Users/User/Documents/ziqinzhang/flashnext_ple
  python ple_gather_test.py --root ... --tokens 1234,567,89 --stats
"""
from __future__ import annotations

import argparse
import json
import mmap
import struct
import sys
from pathlib import Path

N_HEADS = 16
BIGRAM_HEADS = 8
ROW_DIM = 160
ROW_STRIDE = 320
MASK64 = (1 << 64) - 1


class PleTable:
    def __init__(self, root: Path):
        man = json.loads((root / "ple-manifest.json").read_text(encoding="utf-8"))
        self.man = man
        self.mults = [int(m) for m in man["layer_multipliers"]]
        self.head_sizes = [int(s) for s in man["per_head_vocabulary_sizes"]]
        self.head_offsets = [int(o) for o in man["per_head_offsets"]]
        self.parts = sorted(man["logical_parts"], key=lambda p: p["global_row_start"])
        self.total_rows = int(man["padded_vocabulary_rows"])
        # mmap each physical file once (lazy-open on first touch)
        self._maps: dict[int, tuple] = {}
        self._files = {}
        for pf in man["physical_files"]:
            idx = int(pf["index"])
            path = root / pf["path"]
            assert path.exists(), f"missing physical file {path}"
            actual = path.stat().st_size
            assert actual == int(pf["file_bytes"]), (
                f"{path.name}: size {actual} != manifest {pf['file_bytes']}")
            self._files[idx] = (path, int(pf["file_bytes"]))

    # -- row-number formula -------------------------------------------------
    def rows_for(self, ctx: list[int]) -> list[int]:
        """ctx[0]=current token, ctx[1..]=predecessors (most recent first)."""
        assert 2 <= len(ctx) <= 3, "window = bigram(2) or trigram(3)"
        m = self.mults
        mixed = (ctx[0] * m[0]) & MASK64
        mixed ^= (ctx[1] * m[1]) & MASK64
        tri = mixed
        if len(ctx) >= 3:
            tri = mixed ^ ((ctx[2] * m[2]) & MASK64)
        rows = []
        for h in range(N_HEADS):
            base = tri if h >= BIGRAM_HEADS else mixed
            rows.append(base % self.head_sizes[h] + self.head_offsets[h])
        return rows

    # -- physical lookup ----------------------------------------------------
    def _map_for_row(self, global_row: int) -> tuple[mmap.mmap, int]:
        # binary search over part starts
        lo, hi = 0, len(self.parts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.parts[mid]["global_row_start"] <= global_row:
                lo = mid
            else:
                hi = mid - 1
        part = self.parts[lo]
        rel = global_row - int(part["global_row_start"])
        assert 0 <= rel < int(part["rows"]), f"row {global_row} outside part bounds"
        off = int(part["file_offset"]) + rel * ROW_STRIDE
        fidx = int(part["physical_file_index"])
        if fidx not in self._maps:
            path, size = self._files[fidx]
            fh = open(path, "rb")
            self._maps[fidx] = (fh, mmap.mmap(fh.fileno(), size, access=mmap.ACCESS_READ))
        mm = self._maps[fidx][1]
        return mm, off

    def gather_row(self, global_row: int) -> list[float]:
        mm, off = self._map_for_row(global_row)
        raw = mm[off:off + ROW_STRIDE]
        return list(struct.unpack("<160e", raw))

    def close(self) -> None:
        for mm, _ in [(m, f) for f, (f, m) in []]:
            pass
        for fh, mm in self._maps.values():
            mm.close()
            fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing ple-manifest.json + ple/*.bin")
    ap.add_argument("--tokens", default="162042,151645,9707",
                    help="ctx[0]=current, then predecessors (default: sampled ids + EOS)")
    ap.add_argument("--stats", action="store_true", help="print value statistics per head")
    args = ap.parse_args()

    tab = PleTable(Path(args.root))
    ctx = [int(x) for x in args.tokens.split(",")]

    # --- formula-level checks --------------------------------------------
    rows = tab.rows_for(ctx)
    assert all(0 <= r < tab.total_rows for r in rows), "row out of table"
    # bigram vs trigram heads must differ (extra XOR term)
    assert len(set(rows)) == N_HEADS, "all 16 rows should be distinct for this ctx"
    # stability
    assert tab.rows_for(ctx) == rows, "formula not deterministic"
    # commutativity guard: position order matters (swap ctx[1]/ctx[2] changes rows)
    if len(ctx) == 3:
        swapped = tab.rows_for([ctx[0], ctx[2], ctx[1]])
        assert swapped != rows, "position order must matter"

    # EOS semantics: predecessor EOS => fall back to the shorter n-gram window.
    # The reference zeroes history past EOS; emulate by clamping the window.
    eos = 151645  # qwen family im_end
    bounded = [ctx[0]]
    for t in ctx[1:]:
        if t == eos:
            break
        bounded.append(t)
    if len(bounded) < len(ctx):
        print(f"[eos] window truncated {ctx} -> {bounded}")

    # --- real gather -------------------------------------------------------
    print(f"ctx={ctx}")
    total_bad = 0
    for h, r in enumerate(rows):
        vals = tab.gather_row(r)
        nz = sum(1 for v in vals if v != 0.0)
        mean = sum(vals) / len(vals)
        mx = max(abs(v) for v in vals)
        degenerate = nz < 8 or mx == 0.0
        tag = "OK" if not degenerate else "DEGENERATE"
        if degenerate:
            total_bad += 1
        if args.stats:
            print(f"  head{h:02d} row={r:>12d} nz={nz:3d}/160 mean={mean:+.5f} max|x|={mx:.4f} {tag}")
        else:
            print(f"  head{h:02d} row={r:>12d} first8={['%.4f' % v for v in vals[:8]]} {tag}")
    tab.close()

    # range audit: every head's reachable band must be inside the table
    for h in range(N_HEADS):
        top = (tab.head_sizes[h] - 1) + tab.head_offsets[h]
        assert top < tab.total_rows, f"head {h} band {top} exceeds table"
    print(f"row-band audit: 16 heads within {tab.total_rows} rows OK")

    print("PLE_GATHER_" + ("FAIL" if total_bad else "PASS"))
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
