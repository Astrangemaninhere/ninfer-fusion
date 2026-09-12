#!/usr/bin/env python3
"""单变量落地修②：把 NVFP4 GDN snapshot 的 A16 分档阈值 tokens<=3 放宽到 <=16。

目的：让 **verify（W∈[2,16]）与 plain 解码（T=1）走同一激活精度档（A16）**。
现状（已定案）：`variant.cpp:64` 的 kNvfp4TextPolicy = AllowA4，且 snapshot plan 用
`tokens <= 3` 选 A16、否则 A4 ⇒ verify(W=8) 用 FP4 激活（离线实测激活 L2 误差 9.48%、
点积 1.1e-1），而 plain(T=1) 是 A16（0.17% / 1.3e-3）⇒ 两路精度档不同 ⇒ spec≠plain 的主因。

按 P2：先备份、byte-exact 替换、count 断言为 1。
"""

import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp"     # 43 行；路径不符则脚本报错
BAK = "/home/user/fix2_bak"
TARGET = b"tokens <= 3"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    if not path.exists():
        print(f"FAIL: {REL} 不存在；候选：")
        for p in pathlib.Path(BUILD).glob("src/ops/linear/nvfp4/nvfp4_gdn_snapshot_plan.cpp"):
            print("   ", p)
        return 2
    raw = path.read_bytes()
    if raw.count(TARGET) != 1:
        print(f"FAIL: 目标子串出现 {raw.count(TARGET)} 次（期望 1）")
        return 3
    # 替换该行内的 3→16（保持其余字节不变）
    idx = raw.index(TARGET)
    line_start = raw.rfind(b"\n", 0, idx) + 1
    line_end = raw.find(b"\n", idx)
    line = raw[line_start:line_end]
    if line.count(b"tokens <= 3") != 1 or b"tokens <= 16" in line:
        print(f"FAIL: 该行不是唯一的目标形态: {line!r}")
        return 4
    new_line = line.replace(b"tokens <= 3", b"tokens <= 16")
    out = raw[:line_start] + new_line + raw[line_end:]

    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(out)
    print(f"ok: {REL}")
    print(f"   before: {line.decode().strip()}")
    print(f"   after : {new_line.decode().strip()}")
    print(f"   备份  : {BAK}/{path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
