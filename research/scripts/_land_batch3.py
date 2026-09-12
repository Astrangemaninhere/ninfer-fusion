#!/usr/bin/env python3
"""落地三件（都为 CPU 侧、与训练不冲突）：
  A. S28 补完：头文件补 `kv_rowscale_sidecar_apply_from_env` 声明；把定义所在的 .cu 登记进
     ninfer_ops 源列表（紧邻已登记的 gqa_isoquant_row_scale.cu）；恢复 decoder_state.cpp 的调用点。
  B. S51（借用上游 PR #194 的思路）：`ops::silu/sigmoid` 的极端负值归零修复（应用 _collab 的 diff）。
  C. Muse verify 契约修复：`target_verify_batch_impl` 里 lm_head → argmax 之间补
     `apply_final_logit_policy`（契约要求"每个 lm_head logits 产生点"都应用；对 qwen 是编译期 no-op，
     对 Muse（softcap 20 / multiplier 0.196）是必需）。
每处都留 dry-run/幂等检查；不做编译（编译交给下一步）。"""
import pathlib
import re
import subprocess
import sys

R = pathlib.Path("/home/user/ninfer-fusion")

# ---------- A1. 头文件补声明 ----------
H = R / "src/ops/kernel/gqa_isoquant_row_scale_loader.h"
src = H.read_text(encoding="utf-8")
if "kv_rowscale_sidecar_apply_from_env" in src:
    print("A1: 声明已存在")
else:
    anchor = re.search(r"^\[\[nodiscard\]\] inline bool kv_rowscale_sidecar_check\([^;]*;\n",
                       src, re.M | re.S)
    assert anchor, "A1 锚点未找到（sidecar_check 声明）"
    decl = (
        "\n// N3/S28 round 1 engine hook (definition: gqa_isoquant_row_scale_loader.cu).\n"
        "// Reads NINFER_KV_ROWSCALE; unset/empty => false (feature off). Parses and validates\n"
        "// against the loaded model identity, then uploads the table to the device constant.\n"
        "// Any validation failure THROWS (no silent fallback to the baked table).\n"
        "bool kv_rowscale_sidecar_apply_from_env(std::uint32_t model_layers,\n"
        "                                        std::uint32_t model_kv_heads,\n"
        "                                        std::uint32_t model_head_dim,\n"
        "                                        std::uint64_t model_hash);\n"
    )
    src = src[:anchor.end()] + decl + src[anchor.end():]
    H.write_text(src, encoding="utf-8")
    print("A1: 头文件已补声明")

# ---------- A2. CMake 登记 .cu ----------
C = R / "src/CMakeLists.txt"
csrc = C.read_text(encoding="utf-8")
if "gqa_isoquant_row_scale_loader.cu" in csrc:
    print("A2: CMake 已登记")
else:
    line = "  ops/kernel/gqa_isoquant_row_scale.cu\n"
    assert csrc.count(line) == 1, "A2 锚点 %d" % csrc.count(line)
    csrc = csrc.replace(line, line + "  ops/kernel/gqa_isoquant_row_scale_loader.cu\n")
    C.write_text(csrc, encoding="utf-8")
    print("A2: CMake 已登记 .cu")

# ---------- A3. 恢复调用点 ----------
D = R / "src/targets/qwen3_6/impl/state/decoder_state.cpp"
dsrc = D.read_text(encoding="utf-8")
if "kv_rowscale_sidecar_apply_from_env(" in dsrc and "NOT active yet" not in dsrc:
    print("A3: 调用点已在")
else:
    todo = re.search(r"    // S28 \(N3\) row-scale sidecar is NOT active yet.*?\n(?:    //.*\n)+", dsrc, re.S)
    assert todo, "A3 锚点未找到（TODO 注释块）"
    call = ("    // S28: apply the sidecar before any KV write path can observe the table.\n"
            "    (void)ninfer::ops::kv_rowscale_sidecar_apply_from_env(\n"
            "        static_cast<std::uint32_t>(spec.full_attention_layers),\n"
            "        static_cast<std::uint32_t>(spec.kv_heads),\n"
            "        static_cast<std::uint32_t>(spec.attention_head_dim),\n"
            "        0);  // model artifact hash: wired in round 2 (needs options plumbing)\n")
    dsrc = dsrc[:todo.start()] + call + dsrc[todo.end():]
    D.write_text(dsrc, encoding="utf-8")
    print("A3: 调用点已恢复（完全限定名）")

# ---------- B. S51：应用 E8 的 diff ----------
diff = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/E8_s51_nvfp4_silu.diff")
chk = subprocess.run(["patch", "-p1", "--dry-run", "--forward", "-f"], cwd=str(R),
                     stdin=diff.open("rb"), capture_output=True)
if chk.returncode == 0:
    out = subprocess.run(["patch", "-p1", "--forward", "-f"], cwd=str(R),
                         stdin=diff.open("rb"), capture_output=True);
    print("B: S51 已应用" if out.returncode == 0 else "B: S51 应用失败: " + out.stderr.decode()[:200])
else:
    print("B: S51 跳过（dry-run rc=%d，可能已应用）: %s" % (chk.returncode, chk.stderr.decode()[:160]))

# ---------- C. Muse verify 契约 ----------
T = R / "src/targets/qwen3_6/impl/runtime/text_context_impl.h"
tsrc = T.read_text(encoding="utf-8")
if "apply_final_logit_policy(flat_logits" in tsrc:
    print("C: verify policy 已在")
else:
    pat = re.compile(r"( *)ops::linear\(flat_hidden, \*lm_head_, flat_logits, stream\);\n")
    m = pat.search(tsrc)
    assert m, "C 锚点未找到（verify 的 lm_head linear）"
    indent = m.group(1)
    ins = (m.group(0) +
           indent + "// Contract (TextContext::apply_final_logit_policy): every lm_head logits\n" +
           indent + "// production site applies the architecture's final-logit policy. The verify\n" +
           indent + "// path was the one site that did not: a no-op for the qwen family (softcap 0,\n" +
           indent + "// multiplier 1) but REQUIRED for Muse (softcap 20, multiplier 0.196), where\n" +
           indent + "// omitting it made the verifier's argmax differ from plain decoding.\n" +
           indent + "kCfg.apply_final_logit_policy(flat_logits, stream);\n")
    tsrc = tsrc[:m.start()] + ins + tsrc[m.end():]
    T.write_text(tsrc, encoding="utf-8")
    print("C: verify 路径已补 policy")
print("done")
