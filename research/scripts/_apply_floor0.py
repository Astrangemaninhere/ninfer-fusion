#!/usr/bin/env python3
"""实验：把 DFlash2 的接受率地板设为 0（永不再因低接受率而降级/停起草）。

背景：`spec_decision.h:103-107`
    if (stats.drafted_tokens < kDFlash2AcceptanceSample) return false;   // 128
    return accepted/drafted < kDFlash2MinAcceptance;                     // 0.05
⇒ 掉到 5% 以下就"停起草"，实测 87 轮里 64 轮 in_extent=0（大量轮次颗粒无收）。
判据：拿掉地板后，**tok/s 是否提升**（接受率本身预计不变）；若退化则说明"起草不划算"，保留地板。
按 P2：备份 + byte-exact 替换 + count 断言。
"""
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/targets/qwen3_6/impl/runtime/spec_decision.h"
BAK = "/home/user/floor_bak"
OLD = b"inline constexpr double kDFlash2MinAcceptance = 0.05;"
NEW = b"inline constexpr double kDFlash2MinAcceptance = 0.0;  // experiment: never demote on low rate"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    raw = path.read_bytes()
    if raw.count(OLD) != 1:
        print(f"FAIL: 目标出现 {raw.count(OLD)} 次")
        return 2
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(raw.replace(OLD, NEW))
    print("ok: kDFlash2MinAcceptance 0.05 -> 0.0（备份 /home/user/floor_bak/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
