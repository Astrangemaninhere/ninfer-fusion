#!/usr/bin/env python3
"""修 _spec_4way.sh 的 plain_mtp 档：`--spec mtp` 必须带 `--draft-tokens ∈ [1,5]`，
缺了 serve 会打印 usage 并退出（实测 SERVE_FAILED 的日志尾部就是 usage）。
这是已知约束（早前记录过），脚本一直缺这一项。"""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_spec_4way.sh")
src = P.read_text(encoding="utf-8")
old = 'run plain_mtp    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp'
new = '# --spec mtp REQUIRES --draft-tokens in [1,5]; without it serve prints usage and exits\n# (that is exactly the SERVE_FAILED this tier kept producing).\nrun plain_mtp    "$M/qwen3_8_27b_nvfp4.ninfer"          --spec mtp --draft-tokens 3'
if "--draft-tokens 3" in src and "run plain_mtp" in src:
    print("已修过")
else:
    assert src.count(old) == 1, "锚点 %d" % src.count(old)
    src = src.replace(old, new)
    P.write_text(src, encoding="utf-8")
    print("已修：plain_mtp 档补 --draft-tokens 3")
