#!/usr/bin/env python3
"""修 graph_capture_ceiling 的默认值：0（启动全量抓→实测把 family 拖成 eager）→ 16（按需抓）。

依据（实测，MTP k=3 中文）：默认 19.5 tok/s；ceiling=4 → 129.30；ceiling=16 → 130.31；
显式 --no-cuda-graph → 114.99 ⇒ 默认比"关图"还慢 = 最坏组合。
代码依据：program_impl.h:11040/11076 的 `if (graph_capture_ceiling == 0) { capture all }`
分支与其上方注释 "On-demand capture: with a positive ceiling, startup captures only the
segments fully below it; decode extends on growth crossings"。
保留 0 的"全量"语义，只改**默认**。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
BAKDIR = pathlib.Path("/home/user/graph_ceiling_bak")
DEFAULT = 16

SITES = [
    ("apps/cli/options.h", "    std::uint32_t graph_capture_ceiling = 0;"),
    ("src/targets/qwen3_6/impl/runtime/layouts.h", "    std::uint32_t graph_capture_ceiling = 0;"),
    ("src/targets/qwen3_6/impl/runtime/program.h", "    std::uint32_t graph_capture_ceiling = 0;"),
]

for rel, old in SITES:
    p = R / rel
    raw = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    bak = BAKDIR / rel
    bak.parent.mkdir(parents=True, exist_ok=True)
    if not bak.exists():
        bak.write_bytes(raw.encode("utf-8"))
    n = raw.count(old)
    if n == 0:
        print(f"SKIP {rel}（未找到默认行，可能不在此文件）")
        continue
    new = old.replace("= 0;", f"= {DEFAULT};")
    p.write_bytes(raw.replace(old, new).encode("utf-8"))
    print(f"OK  {rel}  默认 0 -> {DEFAULT}（命中 {n} 处）")

print()
for rel, _ in SITES:
    p = R / rel
    if p.exists():
        c = p.read_bytes().decode("utf-8")
        print(f"  回读 {rel}: 默认16 出现 {c.count('graph_capture_ceiling = 16;')} 次，"
              f"默认0 出现 {c.count('graph_capture_ceiling = 0;')} 次")
