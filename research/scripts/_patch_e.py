#!/usr/bin/env python3
"""补丁 E：身份映射新增 nvfp4-dflash2-bf16head（转义修正版）。"""
import pathlib
import re
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6_27b/impl/package.cpp"
p = R / rel
src = p.read_bytes().decode("utf-8")

pattern = (r"(    if \(identity\.model_id == qwen3_8_model_id && identity\.weights_id == \"nvfp4-dflash2\"\) \{\r\n"
           r"        return WeightsProfile::Qwen38Nvfp4DFlash2;\r\n    \}\r\n)")
hits = len(re.findall(pattern, src))
if hits != 1:
    print(f"FAIL 锚点命中 {hits} 次"); sys.exit(3)

# 非 raw 串：\" 产出字面量 "；\r\n 产出 CRLF；\\1 产出反向引用
repl = ("\\1    if (identity.model_id == qwen3_8_model_id &&\r\n"
        "        identity.weights_id == \"nvfp4-dflash2-bf16head\") {\r\n"
        "        return WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head;\r\n    }\r\n")
out = re.sub(pattern, repl, src, count=1)
if '\\"' in out:
    print("FAIL 结果里出现反斜杠引号"); sys.exit(4)
p.write_bytes(out.encode("utf-8"))

chk = (R / rel).read_bytes().decode()
assert '"nvfp4-dflash2-bf16head"' in chk
assert "Qwen38Nvfp4DFlash2Bf16Head" in chk
assert '"nvfp4-dflash2")' in chk            # 原分支仍在
print("OK E 身份映射（原 nvfp4-dflash2 分支保留，新增 bf16head 分支）")
