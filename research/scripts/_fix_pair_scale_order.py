#!/usr/bin/env python3
"""pair_scale must be declared before the scores kernel launch that uses it."""

import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
LCU = "src/ops/launcher/dflash2_selector.cu"

BLOCK = (
    "    float pair_scale = 1.0F;\n"
    "    if (const char* scale = std::getenv(\"NINFER_DF2_PAIR_SCALE\")) {\n"
    "        pair_scale = static_cast<float>(std::atof(scale));\n"
    "    }\n"
)
DEBUG_LINE = ("    const int debug = std::getenv(\"NINFER_DF2SEL\") != nullptr ? 1 : 0;\n")
ANCHOR_LATE = DEBUG_LINE + BLOCK
ANCHOR_EARLY = "    constexpr int kTopK        = kDflash2SelectorTopK;\n"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{LCU}")
    raw = path.read_bytes()
    late = ANCHOR_LATE.encode()
    early = ANCHOR_EARLY.encode()
    if raw.count(late) != 1:
        print(f"FAIL: late anchor occurs {raw.count(late)} times")
        return 2
    if raw.count(early) != 1:
        print(f"FAIL: early anchor occurs {raw.count(early)} times")
        return 2
    raw = raw.replace(late, DEBUG_LINE.encode())
    raw = raw.replace(early, ANCHOR_EARLY.encode() + BLOCK.encode())
    path.write_bytes(raw)
    print("ok: pair_scale declared before its first use")
    return 0


if __name__ == "__main__":
    sys.exit(main())
