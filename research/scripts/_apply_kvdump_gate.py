#!/usr/bin/env python3
"""把 KV dump 的门从 Phase::Prefill 放宽到 (Prefill || Verify)，以便对照 plain(T=1) 与 verify(T=W)
写入的 BF16 K/V（两者同属 Phase::Verify）。

依据：`text_context_impl.h:1007-1008`
    if (kvdump_dir() != nullptr && ph == Phase::Prefill && kvdump_layer_enabled(...))
plain 与 verify 都走 Phase::Verify（A 路：`:810/816` vs `:867/873`），故当前门把它们全排除。
按 P2：备份 + byte-exact + count 断言。
"""
import pathlib
import sys

BUILD = "/home/user/ninfer-fusion"
REL = "src/targets/qwen3_6/impl/runtime/text_context_impl.h"
BAK = "/home/user/kvdump_bak"
OLD = b"    if (kvdump_dir() != nullptr && ph == Phase::Prefill &&\n        kvdump_layer_enabled(\"NINFER_KVDUMP_KV\", fidx)) {"
NEW = b"    if (kvdump_dir() != nullptr && (ph == Phase::Prefill || ph == Phase::Verify) &&\n        kvdump_layer_enabled(\"NINFER_KVDUMP_KV\", fidx)) {"


def main() -> int:
    path = pathlib.Path(f"{BUILD}/{REL}")
    raw = path.read_bytes()
    if raw.count(OLD) != 1:
        print(f"FAIL: 目标出现 {raw.count(OLD)} 次")
        idx = raw.find(b"NINFER_KVDUMP_KV")
        print("   附近:", raw[max(0, idx - 200):idx + 80])
        return 2
    pathlib.Path(BAK).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(BAK) / path.name).write_bytes(raw)
    path.write_bytes(raw.replace(OLD, NEW))
    print("ok: KV dump 门放宽到 Prefill|Verify（备份 /home/user/kvdump_bak/）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
