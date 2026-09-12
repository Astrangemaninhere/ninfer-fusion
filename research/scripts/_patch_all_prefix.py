#!/usr/bin/env python3
"""落 §117 唯一未落地项：支持 `--kv-layer-storage all:<tier>`。

根因（实证）：parse_kv_table 把范围段交给 atoi，"all" → 0 ⇒ all:bf16 只设了第 0 层。
修法：范围段等于 "all" 时展开为 [0, kKvLayerStorageSlots-1]。
行尾自动探测；锚点断言唯一；备份 /home/user/eight_bak/。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/serve/kv_auto_relayout.cpp"
p = R / rel
raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
BAK = pathlib.Path("/home/user/eight_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
if not BAK.exists():
    BAK.write_bytes(raw.encode("utf-8"))
    print("备份 ->", BAK)

anchor = nl.join([
    "            const std::size_t dash = range.find('-');",
    "            int first = 0;",
    "            int last = 0;",
    "            if (dash == std::string_view::npos) {",
])
if raw.count(anchor) != 1:
    print(f"FAIL 锚点命中 {raw.count(anchor)} 次"); sys.exit(3)

repl = nl.join([
    "            const std::size_t dash = range.find('-');",
    "            int first = 0;",
    "            int last = 0;",
    "            if (range == \"all\") {",
    "                // \"all\" 前缀此前会被 atoi 解析成 0 ⇒ `all:bf16` 只设第 0 层，",
    "                // 与\"未设置\"无法区分（_TODO.md 98/117 U3）。显式展开全层。",
    "                first = 0;",
    "                last = static_cast<int>(kKvLayerStorageSlots) - 1;",
    "            } else if (dash == std::string_view::npos) {",
])
p.write_bytes(raw.replace(anchor, repl, 1).encode("utf-8"))

chk = p.read_bytes().decode("utf-8")
assert chk.count('if (range == "all")') == 1
assert chk.count("} else if (dash == std::string_view::npos) {") == 1
print("OK all:<tier> 已支持")
