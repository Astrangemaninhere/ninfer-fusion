#!/usr/bin/env python3
"""新增权重档 Qwen38Nvfp4DFlash2Bf16Head（端点=BF16），实现"lm_head 保留 bf16"。

纯增量补丁：现有 nvfp4 / nvfp4-dflash2 档的行为逐字不变（P2 可干净回滚）。
每处锚点都断言"恰好命中 1 次"，命中数不为 1 即失败退出。
"""
import pathlib
import re
import sys

R = pathlib.Path("/home/user/ninfer-fusion")
edits = []


def patch(path, pattern, repl, label, count=1):
    p = R / path
    src = p.read_text(encoding="utf-8")
    hits = len(re.findall(pattern, src))
    if hits != count:
        print(f"FAIL {label}: 锚点命中 {hits} 次（期望 {count}）")
        sys.exit(3)
    new = re.sub(pattern, repl, src, count=count)
    p.write_text(new, encoding="utf-8")
    edits.append(f"OK  {label}  ({path})")
    print(edits[-1])


# A. 枚举新增档位
patch(
    "src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h",
    r"(    Qwen38Nvfp4DFlash2,\n)(};)",
    r"\1    Qwen38Nvfp4DFlash2Bf16Head,\n\2",
    "A 枚举 WeightsProfile 新增 DFlash2Bf16Head",
)

# B. endpoint_format 新增分支（端点精度 = BF16）
patch(
    "src/targets/qwen3_6_27b/impl/load/bindings.cpp",
    r"(    case WeightsProfile::Qwen38Nvfp4DFlash2:\n        return NumericFormat::FP8_E4M3FN_ROW_BF16S;\n)",
    r"\1    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:\n"
    r"        return NumericFormat::BF16;\n",
    "B endpoint_format 新增 BF16 端点",
)

# C. 文本层绑定 switch：新档与 DFlash2 同组
patch(
    "src/targets/qwen3_6_27b/impl/load/bindings.cpp",
    r"(    case WeightsProfile::Qwen38Nvfp4DFlash2:\n        bind_qwen38_nvfp4_text_layers\(binder, out\);\n        break;\n)",
    r"    case WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head:\n"
    r"\1",
    "C 文本层绑定 switch 新增档位",
)

# D. dflash2 专属绑定：新档一并适用
patch(
    "src/targets/qwen3_6_27b/impl/load/bindings.cpp",
    r"    if \(weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2\) \{\n        const artifact::TensorPlacement dflash2_placement",
    r"    if (weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2 ||\n"
    r"        weights_profile == WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head) {\n"
    r"        const artifact::TensorPlacement dflash2_placement",
    "D dflash2 绑定条件纳入新档",
)

# E. 身份 -> 档位映射
patch(
    "src/targets/qwen3_6_27b/impl/package.cpp",
    r"(    if \(identity\.model_id == qwen3_8_model_id && identity\.weights_id == \"nvfp4-dflash2\"\) \{\n"
    r"        return WeightsProfile::Qwen38Nvfp4DFlash2;\n    \}\n)",
    r"\1    if (identity.model_id == qwen3_8_model_id &&\n"
    r"        identity.weights_id == \"nvfp4-dflash2-bf16head\") {\n"
    r"        return WeightsProfile::Qwen38Nvfp4DFlash2Bf16Head;\n    }\n",
    "E 身份映射新增 nvfp4-dflash2-bf16head",
)

print()
print("补丁应用完成，共", len(edits), "处；备份见 *.orig（若无则用 git/备份目录回滚）")
