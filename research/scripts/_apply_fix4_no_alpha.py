#!/usr/bin/env python3
"""修④：TMA epilogue 不再乘 alpha（与三条兄弟路对齐）。子串定位，不依赖缩进。"""

import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cuh"
BAK = "/home/user/fix4_bak"
PAIRS = [
    (b"silu(gate[0] * alpha) * (up[0] * alpha)", b"silu(gate[0]) * up[0]"),
    (b"silu(gate[1] * alpha) * (up[1] * alpha)", b"silu(gate[1]) * up[1]"),
    (b"silu(gate[2] * alpha) * (up[2] * alpha)", b"silu(gate[2]) * up[2]"),
    (b"silu(gate[3] * alpha) * (up[3] * alpha)", b"silu(gate[3]) * up[3]"),
]
NOTE = (b"    (void)alpha;  // dequant scales are already folded into the codes by the shared\n"
        b"                  // quantize step; the three sibling epilogues (decode/small_t/w4a4)\n"
        b"                  // never rescale here, so this path must not either.\n")


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    raw = path.read_bytes()
    out = raw
    for old, new in PAIRS:
        if out.count(old) != 1:
            print(f"FAIL: {old!r} 出现 {out.count(old)} 次")
            return 2
        out = out.replace(old, new)
    key = b"float alpha,"
    if out.count(key) < 1:
        print("FAIL: 找不到 float alpha, 形参")
        return 3
    pos = out.index(key) + len(key)
    brace = out.index(b"{", pos)
    out = out[:brace + 1] + b"\n" + NOTE + out[brace + 1:]
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(out)
    print("ok: 4 处 silu/up 去 alpha + (void)alpha 注释")
    return 0


if __name__ == "__main__":
    sys.exit(main())
