#!/usr/bin/env python3
"""修 MTP 的 AR 步掩码 off-by-one（零风险）：
`mtp_round.cuh:47` 原为 `ar_valid_columns[offset] = s + 1 < next ? 1 : 0;`
- `next`（`mtp_round.cuh:29-35`）= **下一轮要 draft 的草稿数**（预算与上下文共同限定的 clamp(0,k)）；
- 第 s 步（位置 frontier+s）是否参与 ⟺ `s < next`；
- 原式 `s + 1 < next` ⟺ `s < next - 1`，在 `next == steps`（预算刚好够）时把**最后一个草稿**判为无效 ⇒ 少一个草稿；
- 当 `next > steps` 时两式在 s ≤ steps-1 上**恒等** ⇒ 该修复只在边界改变行为，且方向是"恢复那一个草稿"。"""
import pathlib

P = pathlib.Path("/home/user/ninfer-fusion/src/ops/kernel/mtp_round.cuh")
src = P.read_text(encoding="utf-8")
OLD = "            ar_valid_columns[offset]  = s + 1 < next ? 1 : 0;"
NEW = ("            // Step s is live iff it is inside the next round's draft count. `s + 1 < next`\n"
       "            // dropped the last draft whenever the budget was exactly enough (next == steps);\n"
       "            // the two forms are identical for next > steps.\n"
       "            ar_valid_columns[offset]  = s < next ? 1 : 0;")
if "ar_valid_columns[offset]  = s < next" in src:
    print("已修过")
else:
    assert src.count(OLD) == 1, "锚点 %d" % src.count(OLD)
    src = src.replace(OLD, NEW)
    P.write_text(src, encoding="utf-8")
    print("已修：s + 1 < next -> s < next")
for i, l in enumerate(src.splitlines(), 1):
    if "ar_valid_columns[offset]" in l:
        print("  %4d %s" % (i, l.rstrip()[:110]))
