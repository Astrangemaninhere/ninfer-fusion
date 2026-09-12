#!/usr/bin/env python3
"""新增权重档 Qwen38Nvfp4DFlash2Bf16Head（端点=BF16）。

修正版：
  * 二进制读写，原样保留 CRLF（不改行尾）
  * 替换串不含转义反斜杠（先前 raw string 把 \" 写进了源码）
  * 每处锚点断言恰好命中 1 次；并断言替换后不含 '\\"'
"""
import pathlib
import re
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
BAK = pathlib.Path("/home/user/bf16head_bak")

FILES = [
    "src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h",
    "src/targets/qwen3_6_27b/impl/load/bindings.cpp",
    "src/targets/qwen3_6_27b/impl/package.cpp",
]

# 1) 从备份恢复（清掉上一版带反斜杠的脏改动）
for rel in FILES:
    (R / rel).write_bytes((BAK / rel).read_bytes())
print("已从备份恢复 3 个文件")

# 2) 再打补丁（CRLF 原样）
EDITS = [
    (
        FILES[0],
        r"(    Qwen38Nvfp4DFlash2,\r\n)(\};)",
        "\\1    Qwen38Nvfp4DFlash2Bf16Head,\r\n\\2",
        "A 枚举新增档位",
    ),
    (
        FILES[1],
        r"(    case WeightsProfile::Qwen38Nvfp4DFlash2:\r\n        return NumericFormat::FP8_E4M3FN_ROW_BF16S;\r\n)",
        "\\1    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:\r\n"
        "        return NumericFormat::BF16;\r\n",
        "B endpoint_format 新增 BF16 端点",
    ),
    (
        FILES[1],
        r"(    case WeightsProfile::Qwen38Nvfp4DFlash2:\r\n        bind_qwen38_nvfp4_text_layers\(binder, out\);\r\n        break;\r\n)",
        "    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:\r\n\\1",
        "C 文本层绑定 switch 新增档位",
    ),
    (
        FILES[1],
        r"(    if \(weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2\) \{\r\n        const artifact::TensorPlacement dflash2_placement)",
        "    if (weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2 ||\r\n"
        "        weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head) {\r\n"
        "        const artifact::TensorPlacement dflash2_placement",
        "D dflash2 绑定条件纳入新档",
    ),
    (
        FILES[2],
        r"(    if \(identity\.model_id == qwen3_8_model_id && identity\.weights_id == \"nvfp4-dflash2\"\) \{\r\n"
        r"        return WeightsProfile::Qwen38Nvfp4DFlash2;\r\n    \}\r\n)",
        "\\1    if (identity.model_id == qwen3_8_model_id &&\r\n"
        "        identity.weights_id == \\\"nvfp4-dflash2-bf16head\\\") {\r\n"
        "        return WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head;\r\n    }\r\n",
        "E 身份映射新增档位",
    ),
]

for rel, pattern, repl, label in EDITS:
    p = R / rel
    src = p.read_bytes().decode("utf-8")
    hits = len(re.findall(pattern, src))
    if hits != 1:
        print(f"FAIL {label}: 锚点命中 {hits} 次（期望 1）")
        sys.exit(3)
    out = re.sub(pattern, repl, src, count=1)
    if '\\"' in out:
        print(f"FAIL {label}: 结果里出现 '\\\"'（转义泄漏）")
        sys.exit(4)
    p.write_bytes(out.encode("utf-8"))
    print(f"OK  {label}  ({rel})")

# 3) 回读校验
h = (R / FILES[0]).read_bytes().decode()
b = (R / FILES[1]).read_bytes().decode()
c = (R / FILES[2]).read_bytes().decode()
assert "Qwen38Nvfp4DFlash2Bf16Head," in h
assert "return NumericFormat::BF16;" in b
assert "Qwen38Nvfp4DFlash2Bf16Head)" in b
assert '"nvfp4-dflash2-bf16head"' in c
print("回读校验通过（4 处新档痕迹均在）")
