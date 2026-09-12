#!/usr/bin/env python3
"""The walk launch site did not receive the new `debug` argument."""

import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
LCU = "src/ops/launcher/dflash2_selector.cu"
OLD = b"        batch, steps, top_k);\n"
NEW = b"        batch, steps, top_k, debug);\n"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{LCU}")
    raw = path.read_bytes()
    if raw.count(OLD) != 1:
        print(f"FAIL: anchor occurs {raw.count(OLD)} times")
        return 2
    path.write_bytes(raw.replace(OLD, NEW))
    print("ok: walk launch now passes debug")
    return 0


if __name__ == "__main__":
    sys.exit(main())
