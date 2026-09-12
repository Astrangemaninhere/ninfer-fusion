#!/usr/bin/env python3
"""A3（修正）：当前 decoder_state.cpp 是 S28 之前的干净版（回退所致）⇒ 重新加回
include（顶层）+ 调用（plan_decoder_state 内、用完全限定名）。"""
import pathlib

D = pathlib.Path("/home/user/ninfer-fusion/src/targets/qwen3_6/impl/state/decoder_state.cpp")
src = D.read_text(encoding="utf-8")

# 1) include（必须在文件顶层，别落进 namespace —— S28 原版就是栽在这里）
if "gqa_isoquant_row_scale_loader.h" not in src:
    anchor = '#include "ninfer/ops/entropy_nvfp4_slot.h"\n'
    assert src.count(anchor) == 1, "include 锚点 %d" % src.count(anchor)
    src = src.replace(anchor, anchor + '#include "ops/kernel/gqa_isoquant_row_scale_loader.h"\n')
    print("A3: include 已加（顶层）")
else:
    print("A3: include 已存在")

# 2) 调用
if "kv_rowscale_sidecar_apply_from_env(" not in src:
    anchor = ("DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, "
              "const DecoderStateSpec& spec) {\n    DecoderStateLayout layout;\n")
    assert src.count(anchor) == 1, "函数体锚点 %d" % src.count(anchor)
    call = ("\n    // S28: apply the row-scale sidecar before any KV write path can observe the\n"
            "    // table. Env-gated (NINFER_KV_ROWSCALE); unset => false, nothing happens.\n"
            "    (void)ninfer::ops::kv_rowscale_sidecar_apply_from_env(\n"
            "        static_cast<std::uint32_t>(spec.full_attention_layers),\n"
            "        static_cast<std::uint32_t>(spec.kv_heads),\n"
            "        static_cast<std::uint32_t>(spec.attention_head_dim),\n"
            "        0);  // model artifact hash: round 2 (needs options plumbing)\n")
    src = src.replace(anchor, anchor + call)
    print("A3: 调用点已加（完全限定名）")
else:
    print("A3: 调用点已存在")
D.write_text(src, encoding="utf-8")
print("A3 done")
