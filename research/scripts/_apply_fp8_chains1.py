#!/usr/bin/env python3
"""统一算术（FP8 = 我们 live 的投影路径）：把 T=1 GEMV 的累加链 4 对齐到 SmallT 的 1。

依据：`fp8_config.h` 里
  - T=1 GEMV   : Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>   ← 累加链 4
  - T∈[2,16] ST: Fp8SmallTSchedule<8, 2, kValuesPerLane, kTokenTile, 1, …>  ← 累加链 1
两侧其余结构（lane→K 映射、scale 分组、归约、epilogue）不同文件实现，是否**逐位**一致要由实测裁决
（判据：`_ga_check` 里 zh/num 的 spec-vs-plain 首次偏离是否消失/后移；代价：decode tok/s）。
按 P2：备份 + count 断言 + 全量替换。
"""
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/ops/linear/fp8/fp8_config.h"
BAK = "/home/user/fp8chain_bak"
OLD = b"Fp8GemvSchedule<8, 2, 8, 4,"
NEW = b"Fp8GemvSchedule<8, 2, 8, 1,"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    raw = path.read_bytes()
    n = raw.count(OLD)
    if n == 0:
        print("FAIL: 目标 0 次")
        return 2
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(raw.replace(OLD, NEW))
    print(f"ok: {n} 处 Fp8GemvSchedule 累加链 4 -> 1（备份 /home/user/fp8chain_bak/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
