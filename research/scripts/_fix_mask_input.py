#!/usr/bin/env python3
"""修 M2（训练/推理输入模式 skew）：`train_dflash2.py` 的 block 用真 token，而引擎的
`prepare_masked_block(anchors, frontiers, valid, Config::mask_token, ids, positions)` 喂的是
`[anchor, mask×(B-1)]` ⇒ 训练时 mask id 的 embedding 从未被优化，推理却被喂 7 个 mask
（A2 实测：mask 口径命中率比真 token 口径低 5.5 倍，训练语料里 mask id 出现 0/4468 次）。

改法：block 第 0 列保留 anchor（= tok[a]，与引擎 ids[0]=anchor 一致），1..B-1 列填 mask id；
mask id 取**引擎的实际值** 248077（`src/targets/qwen3_6_27b/impl/config.h:106`）。"""
import pathlib
import py_compile

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py")
src = P.read_text(encoding="utf-8")

OLD_CONST = "BLOCK_SIZE = 8            # verify width incl. anchor; drafts per round = B-1"
NEW_CONST = (
    "BLOCK_SIZE = 8            # verify width incl. anchor; drafts per round = B-1\n"
    "# The engine feeds block column 0 = anchor and columns 1..B-1 = this mask id\n"
    "# (`ops::prepare_masked_block(anchors, frontiers, valid, Config::mask_token, ...)`,\n"
    "#  dflash2_impl.h:192) -- it must match `mask_token` in\n"
    "# `src/targets/qwen3_6_27b/impl/config.h:106`, otherwise the draft is trained on an\n"
    "# input pattern it never sees at inference (A2 measured the skew: mask-input hit rate\n"
    "# 0.031 vs true-token 0.171, and the mask id never appears in the corpus: 0/4468).\n"
    "MASK_ID = 248077"
)
if "MASK_ID = 248077" in src:
    print("常量已加")
else:
    assert src.count(OLD_CONST) == 1, "常量锚点 %d" % src.count(OLD_CONST)
    src = src.replace(OLD_CONST, NEW_CONST)
    print("已加 MASK_ID 常量")

OLD_BLOCK = """            block = tok[a:a + B].unsqueeze(0)         # (1, B)"""
NEW_BLOCK = """            # Match inference exactly: column 0 is the anchor, columns 1..B-1 are the
            # mask the engine will feed (train_dflash2.py used true tokens here before,
            # which left the mask embedding untrained).
            block = tok[a:a + B].clone().unsqueeze(0)  # (1, B)
            block[0, 1:] = MASK_ID"""
if "block[0, 1:] = MASK_ID" in src:
    print("block 构造已修")
else:
    assert src.count(OLD_BLOCK) == 1, "block 锚点 %d" % src.count(OLD_BLOCK)
    src = src.replace(OLD_BLOCK, NEW_BLOCK)
    print("已修 block 构造（mask 输入）")

P.write_text(src, encoding="utf-8")
py_compile.compile(str(P), doraise=True)
print("py_compile OK")
