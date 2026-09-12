#!/usr/bin/env python3
"""Fix the dflash2 walk's argmax.

`value` is overwritten by the fmax reduction and only then compared against `best`.
Lane 0 ends the reduction holding the global maximum, so `equal` is true for lane 0 on
essentially every step and the `min(lane)` tie-break always returns rank 0 — i.e. the
walk silently degenerates into "take the per-step unary argmax" and the trained edge
term (measured pairspan 4.9..12.4, strongly decisive) never influences the choice. That
is why drafts collapse into runs of one token and every position past the first fails.

Compare each lane's OWN pre-reduction score, which restores the contract's
"lowest-rank argmax_c E_i[j_(i-1),c]".
"""

import difflib
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
OUT = "/mnt/c/Users/User/Documents/ziqinzhang/_collab/F4_df2_walk_argmax_fix.diff"
KERN = "src/ops/kernel/dflash2_selector.cuh"

OLD = """#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_xor_sync(Mask, value, offset));
        }
        const float best              = __shfl_sync(Mask, value, 0);
        const bool equal              = lane < K && value == best;
"""

NEW = """        // Keep this lane's own score: the reduction below overwrites `value`, and
        // comparing the reduced partial maxima lets lane 0 win almost every step, which
        // silently discards the edge term and degenerates the walk into a per-step unary
        // argmax.
        const float own_value = value;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_xor_sync(Mask, value, offset));
        }
        const float best              = __shfl_sync(Mask, value, 0);
        const bool equal              = lane < K && own_value == best;
"""


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{KERN}")
    raw = path.read_bytes()
    old = OLD.encode()
    if raw.count(old) != 1:
        print(f"FAIL: anchor occurs {raw.count(old)} times")
        return 2
    new = NEW.encode()
    path.write_bytes(raw.replace(old, new))
    chunks = "".join(difflib.unified_diff(
        raw.decode().splitlines(keepends=True), new.decode().splitlines(keepends=True),
        f"a/{KERN}", f"b/{KERN}", n=4))
    pathlib.Path(OUT).write_text(chunks, encoding="utf-8")
    print("ok: walk argmax now compares each lane's own score")
    print(f"diff -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
