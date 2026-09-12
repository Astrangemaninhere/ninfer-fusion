#!/usr/bin/env python3
"""variant.cpp：7 处 switch 的 DF2 分支后并入新档（文本层精度与 DF2 相同）。"""
import pathlib
import re
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
rel = "src/targets/qwen3_6_27b/impl/variant.cpp"
p = R / rel

# 备份
BAK = pathlib.Path("/home/user/bf16head_bak") / rel
BAK.parent.mkdir(parents=True, exist_ok=True)
BAK.write_bytes(p.read_bytes())
print("已备份 ->", BAK)

src = p.read_bytes().decode("utf-8")
pattern = r"    case WeightsProfile::Qwen38Nvfp4DFlash2:\r\n"
n = len(re.findall(pattern, src))
print(f"DF2 case 命中 {n} 处（期望 7）")
if n != 7:
    sys.exit(3)

out = re.sub(pattern,
             "    case WeightsProfile::Qwen38Nvfp4DFlash2:\r\n"
             "    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:\r\n",
             src)
if '\\"' in out:
    sys.exit(4)
p.write_bytes(out.encode("utf-8"))

chk = p.read_bytes().decode()
print("新档 case 数量 =", chk.count("case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:"))
assert chk.count("case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:") == 7
print("OK variant.cpp 7 处已并入")
