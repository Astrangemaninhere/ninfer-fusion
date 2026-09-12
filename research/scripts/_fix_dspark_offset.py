#!/usr/bin/env python3
"""修 dspark 的 block 列错位（根因）：

训练约定（`train_dspark.py:168-169, 428-432`）：
    block_pos = arange(a, a+B)          # 第 0 列 = anchor（位置 a，已提交的 token）
    logits    = lm_head(out[:, 1:])     # 输出只用 columns 1..k
    teacher_idx[p-1] = tok[a+p]         # 第 p 列预测**它自己那一列**的 token
⇒ 推理时必须取 columns **1..k**（跳过 anchor 列）。

引擎（`dflash_impl.h:448-453`）现在是：
    std::size_t source_column_offset = 0;
    if constexpr (!Config::bf16_weights) { source_column_offset = 1; }   // 注释："Legacy DFlash keeps the anchor column out of the proposal rows"
⇒ **bf16 权重时 offset=0（含 anchor 列）**，与训练约定冲突。我们的 dspark 草稿正是 bf16（审计 55/55 逐字节等于 checkpoint）
⇒ 整块草稿错位一列 ⇒ p(pos0) 崩。

修法：按训练约定，**所有** DFlash 草稿都跳过 anchor 列（offset 恒 1）。保留常量名以便日后配置化。"""
import pathlib

P = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash_impl.h")
src = P.read_text(encoding="utf-8")
OLD = """        std::size_t source_column_offset = 0;
        if constexpr (!Config::bf16_weights) {
            // Legacy DFlash keeps the anchor column out of the proposal rows.
            source_column_offset = 1;
        }"""
NEW = """        // The trainer pairs block column p (position a+p) with token x_{a+p} and only
        // consumes `out[:, 1:]` (train_dspark.py:168-169,428-432): column 0 is the anchor
        // and never part of the proposal. Taking columns 0..k-1 here therefore shifted the
        // whole draft block by one column for bf16 drafts (the dspark artifact is bf16),
        // which is what collapsed its position-0 acceptance. Skip the anchor column always;
        // the offset stays a named constant so it can become an artifact property later.
        constexpr std::size_t source_column_offset = 1;"""
if "The trainer pairs block column p" in src:
    print("已修过")
else:
    assert src.count(OLD) == 1, "锚点 %d" % src.count(OLD)
    src = src.replace(OLD, NEW)
    P.write_text(src, encoding="utf-8")
    print("已修：DFlash 草稿恒跳过 anchor 列（offset=1）")
for i, l in enumerate(src.splitlines(), 1):
    if "source_column_offset" in l:
        print("  %4d %s" % (i, l.rstrip()[:120]))
