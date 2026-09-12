#!/usr/bin/env python3
"""解锁 BF16 全词表头：把 (248320,5120) 与 (131072,5120) 加入 bf16 dispatch 的
dflash2_problem 白名单，使 BF16 的 text/output_head 与 text/draft_head 可走 launch_bf16_mma。

动机（非性能）：这是 (c) 官方 bf16 权重到手之前，唯一能拿到
"head+embedding 量化误差"精确且非循环参考的路径；也是 "lm_head 保留 bf16" 的最后一格。
注意 n 可被 64 整除、k 可被 128 整除（248320/64=3880、131072/64=2048、5120/128=40），
故通用 64x128x64 MMA 分块可覆盖；代价是 t=1 时比 FP8 GEMV 慢——仅供实验档使用。
"""
import pathlib
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/ops/linear/bf16/bf16_dispatch.cpp"
p = R / rel

BAK = pathlib.Path("/home/user/bf16head_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(p.read_bytes())
print("已备份 ->", BAK)

raw = p.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"
print("行尾:", "CRLF" if nl != "\n" else "LF")

anchor = nl.join([
    "                                 (n == 34816 && k == 5120) || (n == 5120 && k == 17408) ||",
    "                                 (n == 256 && k == 5120);",
    "",
])
if raw.count(anchor) != 1:
    print(f"FAIL 锚点命中 {raw.count(anchor)} 次"); sys.exit(3)

repl = nl.join([
    "                                 (n == 34816 && k == 5120) || (n == 5120 && k == 17408) ||",
    "                                 (n == 256 && k == 5120) ||",
    "                                 // 全词表/短表 vocabulary 头：BF16 档的 text/output_head",
    "                                 // 与 text/draft_head（供 head 量化误差的非循环参考）。",
    "                                 (n == 248320 && k == 5120) || (n == 131072 && k == 5120);",
    "",
])
p.write_bytes(raw.replace(anchor, repl, 1).encode("utf-8"))

chk = p.read_bytes().decode("utf-8")
assert chk.count("n == 248320 && k == 5120") == 1
assert chk.count("n == 131072 && k == 5120") == 1
print("OK 白名单已补两形（248320x5120 / 131072x5120）")
