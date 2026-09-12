#!/usr/bin/env python3
"""Verify the --target-shift edits: file compiles, default path is unchanged."""
import py_compile
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py")
py_compile.compile(str(P), doraise=True)
print("py_compile OK")

src = P.read_text(encoding="utf-8")
for needle in ("--target-shift", "t16_ids = ", "t16_vals = ", "globals()['OUT_DIR']"):
    print("--- %s ---" % needle)
    for line in src.splitlines():
        if needle in line:
            print("   ", line.strip()[:120])
