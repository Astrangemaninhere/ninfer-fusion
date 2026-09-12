#!/usr/bin/env python3
"""Make the edge-term weight adjustable at runtime (NINFER_DF2_PAIR_SCALE, default 1.0).

After the walk fix the edge term actually decides the path, and it drives the walk to rare
tokens. This knob separates "the edge term is mis-scaled" from "the edge tables themselves
are weak": sweep the scale and watch acceptance. Gate is an env read in the launcher, so
the default path is bit-identical to the contract arithmetic.
"""

import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
KERN = "src/ops/kernel/dflash2_selector.cuh"
LCU = "src/ops/launcher/dflash2_selector.cu"

EDITS = [
    (KERN,
     "    const std::int32_t* __restrict__ anchors, float* __restrict__ scores, int vocab, int batch,\n"
     "    int steps, int top_k, int columns) {\n",
     "    const std::int32_t* __restrict__ anchors, float* __restrict__ scores, int vocab, int batch,\n"
     "    int steps, int top_k, int columns, float pair_scale) {\n"),
    (KERN,
     "    scores[dflash2_selector_score_offset(b, batch, s, steps, p, c, top_k)] =\n"
     "        pair + shared_unary[c];\n",
     "    scores[dflash2_selector_score_offset(b, batch, s, steps, p, c, top_k)] =\n"
     "        pair * pair_scale + shared_unary[c];\n"),
    (LCU,
     "    const int debug = std::getenv(\"NINFER_DF2SEL\") != nullptr ? 1 : 0;\n",
     "    const int debug = std::getenv(\"NINFER_DF2SEL\") != nullptr ? 1 : 0;\n"
     "    float pair_scale = 1.0F;\n"
     "    if (const char* scale = std::getenv(\"NINFER_DF2_PAIR_SCALE\")) {\n"
     "        pair_scale = static_cast<float>(std::atof(scale));\n"
     "    }\n"),
    (LCU,
     "        static_cast<const std::int32_t*>(anchors.data), static_cast<float*>(scores.data),\n"
     "        unary_logits.ne[0], batch, steps, top_k, columns);\n",
     "        static_cast<const std::int32_t*>(anchors.data), static_cast<float*>(scores.data),\n"
     "        unary_logits.ne[0], batch, steps, top_k, columns, pair_scale);\n"),
]


def main() -> int:
    edited = {}
    for rel in {p for p, _, _ in EDITS}:
        edited[rel] = pathlib.Path(f"{BUILD}/{rel}").read_bytes()
    for rel, old, new in EDITS:
        raw = edited[rel]
        if raw.count(old.encode()) != 1:
            print(f"FAIL {rel}: anchor occurs {raw.count(old.encode())} times")
            print("   head:", old[:80])
            return 2
        edited[rel] = raw.replace(old.encode(), new.encode())
    for rel, raw in edited.items():
        pathlib.Path(f"{BUILD}/{rel}").write_bytes(raw)
    print("ok: pair scale is now adjustable via NINFER_DF2_PAIR_SCALE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
