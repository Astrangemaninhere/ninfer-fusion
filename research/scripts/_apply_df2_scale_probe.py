#!/usr/bin/env python3
"""Add an env-gated scale probe to the dflash2 selector walk.

Question it settles: does the edge (pair) term carry any weight against the unary?
Observed drafts degenerate into runs of one token on prose, which is what a walk that
just follows the per-step unary argmax looks like. This prints, per step, the winner's
E = u + pair decomposed, plus the unary row's span (max-min) as the scale to compare
the pair term against. Zero effect unless NINFER_DF2SEL is set.
"""

import difflib
import hashlib
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
BAK = "/home/user/df2head_bak"
OUT = "/mnt/c/Users/User/Documents/ziqinzhang/_collab/F2_df2_scale_probe.diff"
KERN = "src/ops/kernel/dflash2_selector.cuh"
LCU = "src/ops/launcher/dflash2_selector.cu"

EDITS = []


def add(path, old, new, note):
    EDITS.append((path, old.encode(), new.encode(), note))


add(KERN,
    "                                             const float* __restrict__ scores,\n"
    "                                             std::int32_t* __restrict__ drafts,\n"
    "                                             const SamplingConfig* __restrict__ configs,\n"
    "                                             std::int32_t* __restrict__ out_ids,\n"
    "                                             float* __restrict__ out_probs, int batch,\n"
    "                                             int steps, int top_k) {\n",
    "                                             const float* __restrict__ scores,\n"
    "                                             const float* __restrict__ unary,\n"
    "                                             std::int32_t* __restrict__ drafts,\n"
    "                                             const SamplingConfig* __restrict__ configs,\n"
    "                                             std::int32_t* __restrict__ out_ids,\n"
    "                                             float* __restrict__ out_probs, int batch,\n"
    "                                             int steps, int top_k, int debug) {\n",
    "walk kernel: unary + debug params")

add(KERN,
    "        // Publish this step's candidate distribution: softmax over the raw row\n",
    "        // Env-gated (NINFER_DF2SEL): decompose the winner into unary and edge, with\n"
    "        // the unary row's span as the scale to judge the edge term against.\n"
    "        if (debug != 0 && b == 0 && lane == 0) {\n"
    "            const float e_chosen = scores[dflash2_selector_score_offset(b, batch, s, steps,\n"
    "                                                                       pred, chosen, top_k)];\n"
    "            const float u_chosen =\n"
    "                unary[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)];\n"
    "            float u_hi = -CUDART_INF_F;\n"
    "            float u_lo = CUDART_INF_F;\n"
    "            for (int c = 0; c < K; ++c) {\n"
    "                const float u =\n"
    "                    unary[dflash2_selector_candidate_offset(b, batch, s, steps, c)];\n"
    "                u_hi = fmaxf(u_hi, u);\n"
    "                u_lo = fminf(u_lo, u);\n"
    "            }\n"
    "            printf(\"[df2sel] s=%d pred=%d chosen=%d tok=%d E=%.3f u=%.3f pair=%.3f \"\n"
    "                   \"uspan=%.3f\\n\",\n"
    "                   s, pred, chosen,\n"
    "                   candidates[dflash2_selector_candidate_offset(b, batch, s, steps, chosen)],\n"
    "                   e_chosen, u_chosen, e_chosen - u_chosen, u_hi - u_lo);\n"
    "        }\n"
    "\n"
    "        // Publish this step's candidate distribution: softmax over the raw row\n",
    "walk kernel: print E/u/pair/uspan")

add(LCU,
    "#include <cstdint>\n#include <stdexcept>\n",
    "#include <cstdint>\n#include <cstdlib>\n#include <stdexcept>\n",
    "launcher: include cstdlib")

add(LCU,
    "    dflash2_selector_walk_kernel<<<batch, 32, 0, stream>>>(\n"
    "        static_cast<const std::int32_t*>(candidates.data),\n"
    "        static_cast<const float*>(scores.data), static_cast<std::int32_t*>(drafts.data), configs,\n",
    "    const int debug = std::getenv(\"NINFER_DF2SEL\") != nullptr ? 1 : 0;\n"
    "    dflash2_selector_walk_kernel<<<batch, 32, 0, stream>>>(\n"
    "        static_cast<const std::int32_t*>(candidates.data),\n"
    "        static_cast<const float*>(scores.data),\n"
    "        static_cast<const float*>(unary.data),\n"
    "        static_cast<std::int32_t*>(drafts.data), configs,\n",
    "launcher: pass unary and the debug flag")


def main() -> int:
    files = sorted({p for p, _, _, _ in EDITS})
    originals = {}
    for rel in files:
        raw = pathlib.Path(f"{BUILD}/{rel}").read_bytes()
        originals[rel] = raw
        pathlib.Path(f"{BAK}/{rel.replace('/', '__')}").write_bytes(raw)

    edited = dict(originals)
    for rel, old, new, note in EDITS:
        raw = edited[rel]
        if raw.count(old) != 1:
            print(f"FAIL {rel}: anchor occurs {raw.count(old)} times ({note})")
            print("     head:", old[:90])
            return 2
        edited[rel] = raw.replace(old, new)
        print(f"ok   {rel:<48} {note}")

    chunks = []
    for rel in files:
        before = originals[rel].decode("utf-8", "replace").splitlines(keepends=True)
        after = edited[rel].decode("utf-8", "replace").splitlines(keepends=True)
        chunks.append("".join(difflib.unified_diff(before, after, f"a/{rel}", f"b/{rel}", n=3)))
        pathlib.Path(f"{BUILD}/{rel}").write_bytes(edited[rel])
    pathlib.Path(OUT).write_text("".join(chunks), encoding="utf-8")
    print(f"diff -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
