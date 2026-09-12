#!/usr/bin/env python3
"""A2 建议保留 A/B 能力：把 mask 输入做成默认开启、可用 `--no-mask-block` 关闭，
这样"修好的配方"与"旧配方"能在同一 ckpt 上对比（复现 §10-1 的决胜实验）。"""
import pathlib
import py_compile
import re

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py")
src = P.read_text(encoding="utf-8")

# 1) 参数
anchor = "    ap.add_argument('--target-shift', type=int, default=0)"
add = anchor + """
    # Train on the engine's actual input pattern ([anchor, mask x (B-1)]); pass
    # --no-mask-block to reproduce the old all-true-token recipe for A/B.
    ap.add_argument('--mask-block', dest='mask_block', action='store_true', default=True)
    ap.add_argument('--no-mask-block', dest='mask_block', action='store_false')"""
if "--no-mask-block" in src:
    print("开关已存在")
else:
    assert src.count(anchor) == 1, "参数锚点 %d" % src.count(anchor)
    src = src.replace(anchor, add)
    print("已加 --mask-block/--no-mask-block")

# 2) block 构造：按开关决定是否填 mask
old = """            block = tok[a:a + B].clone().unsqueeze(0)  # (1, B)
            block[0, 1:] = MASK_ID"""
new = """            block = tok[a:a + B].clone().unsqueeze(0)  # (1, B)
            if args.mask_block:
                block[0, 1:] = MASK_ID"""
if "if args.mask_block:" in src:
    print("block 构造已按开关")
else:
    assert src.count(old) == 1, "block 锚点 %d" % src.count(old)
    src = src.replace(old, new)
    print("已改 block 构造（受开关控制）")

P.write_text(src, encoding="utf-8")
py_compile.compile(str(P), doraise=True)
print("py_compile OK")
for i, l in enumerate(src.splitlines(), 1):
    if "mask_block" in l or "MASK_ID" in l or "target-shift" in l:
        print("  %4d %s" % (i, l.rstrip()[:120]))
