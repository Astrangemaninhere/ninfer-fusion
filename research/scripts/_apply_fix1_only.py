#!/usr/bin/env python3
"""单变量落地修①（两路逐字一致的共同核心）：
`gdn_conv.cuh:99` 把 conv 第 4 抽头的累加值先 bf16 化，与 publish 侧（只落 bf16(p)）同源。

B 路已证：发布侧逐位不变（bf16(bf16(p))==bf16(p)），修后 fused conv 与 materialized post conv
（`nvfp4_gdn_snapshot_post.cu:96-100`）逐表达式相同 ⇒ 对同一 p 逐位等价。
"""
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/ops/gdn_input_proj/gdn_conv.cuh"
BAK = "/home/user/fix1_bak"
OLD = b"= projected[token];"
NEW = b"= __bfloat162float(__float2bfloat16_rn(projected[token]));"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    raw = path.read_bytes()
    if raw.count(OLD) != 1:
        print(f"FAIL: 目标出现 {raw.count(OLD)} 次（期望 1）")
        idx = raw.find(b"projected[token]")
        if idx >= 0:
            print("   上下文:", raw[max(0, idx - 80):idx + 40])
        return 2
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(raw.replace(OLD, NEW))
    print("ok: gdn_conv.cuh:99 修① 已落地（备份 /home/user/fix1_bak/gdn_conv.cuh）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
