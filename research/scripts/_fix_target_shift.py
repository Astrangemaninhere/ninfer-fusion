#!/usr/bin/env python3
"""修 dflash2 训练目标偏移一行（第二处同族 bug）：
实证：`ids16[t]` 是「位置 t 的 next-token 分布」（P(ids16[t]==tok[t+1])=0.2904，P(==tok[t])=0.0008，
样本 3832）；而 `head()` 的 `out[:, 1:]` 让 block 槽位 i 对应位置 a+1+i ⇒ 其正确目标行是 `ids16[a+i]`
⇒ `--target-shift` 必须为 0。默认 1 取 `ids16[a+1+i]`（位置 a+2+i 的 token）⇒ 目标晚一行。
把默认改成 0，并在帮助文本里写明口径依据（避免以后又改回去）。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py")
src = P.read_text(encoding="utf-8")
OLD = "    ap.add_argument('--target-shift', type=int, default=1)"
NEW = ("    # ids16[t] is the teacher's NEXT-Token distribution at position t (measured on the real\n"
       "    # cache: P(ids16[t]==tok[t+1])=0.2904 vs P(ids16[t]==tok[t])=0.0008, n=3832), so the\n"
       "    # teacher row for the token AT position a+1+i is ids16[a+i]. head() already slices\n"
       "    # out[:, 1:], i.e. block slot i sits at position a+1+i => the aligned shift is 0.\n"
       "    # The old default of 1 trained every slot one position ahead (the draft predicted\n"
       "    # a+2+i), which is exactly the acceptance-killing mismatch this argument exists for.\n"
       "    ap.add_argument('--target-shift', type=int, default=0)")
if "the aligned shift is 0" in src:
    print("已修过")
else:
    assert src.count(OLD) == 1, "锚点 %d" % src.count(OLD)
    src = src.replace(OLD, NEW)
    P.write_text(src, encoding="utf-8")
    print("已修：--target-shift 默认 1 -> 0（附口径实测依据）")
import py_compile  # noqa: E402
py_compile.compile(str(P), doraise=True)
print("py_compile OK")
