#!/usr/bin/env python3
"""Extend the selector probe: measure the edge term's spread ACROSS candidates.

The first probe showed |pair| is large (up to 6.4) yet the walk never leaves candidate
rank 0, while the unary row span is only ~1.5-2.8. Either the pair term is nearly flat
across c (so it cannot reorder anything), or it varies but correlates with u. Print, per
step: pair span over c, the E-argmax rank, and the u-argmax rank.
"""

import difflib
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
OUT = "/mnt/c/Users/User/Documents/ziqinzhang/_collab/F3_df2_pair_spread.diff"
KERN = "src/ops/kernel/dflash2_selector.cuh"

OLD = """            float u_hi = -CUDART_INF_F;
            float u_lo = CUDART_INF_F;
            for (int c = 0; c < K; ++c) {
                const float u =
                    unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)];
                u_hi = fmaxf(u_hi, u);
                u_lo = fminf(u_lo, u);
            }
            printf("[df2sel] s=%d pred=%d chosen=%d tok=%d E=%.3f u=%.3f pair=%.3f "
                   "uspan=%.3f\\n",
                   s, pred, chosen,
                   candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)],
                   e_chosen, u_chosen, e_chosen - u_chosen, u_hi - u_lo);
"""

NEW = """            float u_hi = -CUDART_INF_F;
            float u_lo = CUDART_INF_F;
            float pair_hi = -CUDART_INF_F;
            float pair_lo = CUDART_INF_F;
            float e_hi = -CUDART_INF_F;
            int e_arg = -1;
            int u_arg = -1;
            for (int c = 0; c < K; ++c) {
                const float u =
                    unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)];
                const float pair_c =
                    scores[dflash2_selector_score_offset(b, batch, s, steps, pred, c, top_k)] - u;
                u_lo = fminf(u_lo, u);
                pair_hi = fmaxf(pair_hi, pair_c);
                pair_lo = fminf(pair_lo, pair_c);
                if (c == 0 || u > u_hi) {
                    u_hi = u;
                    u_arg = c;
                }
                if (c == 0 || u + pair_c > e_hi) {
                    e_hi = u + pair_c;
                    e_arg = c;
                }
            }
            printf("[df2sel] s=%d pred=%d chosen=%d tok=%d E=%.3f u=%.3f pair=%.3f "
                   "uspan=%.3f pairspan=%.3f Earg=%d Uarg=%d\\n",
                   s, pred, chosen,
                   candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)],
                   e_chosen, u_chosen, e_chosen - u_chosen, u_hi - u_lo, pair_hi - pair_lo,
                   e_arg, u_arg);
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
        f"a/{KERN}", f"b/{KERN}", n=3))
    pathlib.Path(OUT).write_text(chunks, encoding="utf-8")
    print("ok: probe now prints pairspan / Earg / Uarg")
    print(f"diff -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
