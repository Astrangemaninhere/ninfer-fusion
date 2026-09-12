#!/usr/bin/env python3
"""The shared dflash2_impl.h needs `Config::draft_head_rows` for EVERY variant's
DFlash2Config. Each variant.h already declares Variant::draft_head_rows = 131072, so the
value is not in question; only the per-target config constant was missing."""

import hashlib
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
BAK = "/home/user/df2head_bak"

ADD = ("    // Rows of the checkpoint's dedicated proposal head (`text/draft_head`). Its\n"
       "    // rows are shortlist entries: only the global ids reached through\n"
       "    // `text/draft_head_token_ids` may index the selector codebooks (contract 6.1).\n"
       "    static constexpr int draft_head_rows = 131072;\n")

TARGETS = [
    ("src/targets/qwen3_6_35b_a3b/impl/config.h",
     "    static constexpr int selector_top_k  = 16;\n"),
    ("src/targets/muse_glimmer_30b/impl/config.h",
     "    static constexpr int selector_top_k  = 0;\n"),
]


def main() -> int:
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    for rel, anchor in TARGETS:
        path = pathlib.Path(f"{BUILD}/{rel}")
        raw = path.read_bytes()
        (pathlib.Path(BAK) / rel.replace("/", "__")).write_bytes(raw)
        if b"draft_head_rows" in raw:
            print(f"skip {rel}: already has draft_head_rows")
            continue
        if raw.count(anchor.encode()) != 1:
            print(f"FAIL {rel}: anchor occurs {raw.count(anchor.encode())} times")
            return 2
        path.write_bytes(raw.replace(anchor.encode(), anchor.encode() + ADD.encode()))
        print(f"ok   {rel}: +draft_head_rows (md5 {hashlib.md5(raw).hexdigest()[:12]} "
              f"-> {hashlib.md5((anchor + ADD).encode()).hexdigest()[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
