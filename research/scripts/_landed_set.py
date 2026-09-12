#!/usr/bin/env python3
"""One authoritative list of what this build contains (the measurement's provenance)."""
import datetime
import hashlib
import pathlib
import subprocess

R = pathlib.Path("/home/user/ninfer-fusion")
ITEMS = [
    ("补丁 A", "src/targets/qwen3_6/impl/runtime/program_impl.h",
     "DFlash2 decode ingress 漏填 state_source/destination_slots（M_patchA_*.diff 已应用）"),
    ("E2/S44", "src/serve/request_log.cpp", "接受率计数标签 spec_drafted/spec_accepted/spec_accept_rate"),
    ("E2/S44", "apps/cli/main.cpp", "CLI accepted-by-pos 直方图"),
    ("E4/S46", "src/serve/kv_cold_policy.h", "冷窗 F3b：空闲窗不进 EWMA + 置信度门"),
    ("E4/S46", "src/serve/kv_auto_relayout.cpp", "touched 以强制重编（头文件依赖未跟踪）"),
    ("S45d/E3", "src/ops/launcher/gqa_attention_prefill.cu", "128 几何逐臂 nvfp4 守卫（256 下逐字节等价）"),
    ("S45c/E3", "src/serve/serve_options.cpp", "--spec usage 文本补 dflash2|auto"),
    ("S45c/E3", "apps/cli/options.cpp", "同上（CLI 侧）"),
    ("S48/E6", "src/targets/qwen3_6/impl/runtime/dflash_impl.h", "dspark verify 位置表 k→k+1"),
    ("S50/E7", "src/targets/qwen3_6/impl/runtime/logical_kv_store.h", "KV 覆盖下界语义 ensure_mapped_to_tokens"),
    ("S50/E7", "src/targets/qwen3_6/impl/runtime/program.h", "声明改名 ensure_sequence_kv_mapped"),
    ("S50/E7", "src/targets/qwen3_6/impl/runtime/program_impl.h", "14 处调用点改名（参数逐字节不变）"),
    ("S50/E7", "tests/targets/qwen3_6/test_context_store.cpp", "测试侧同步改名"),
]

out = ["# 本趟编译的落地集（provenance，%s）" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
       "判读用：测量出来的每个数字都归属于这一组文件哈希。", "",
       "| 补丁 | 文件 | md5 | 说明 |", "|---|---|---|---|"]
for tag, rel, note in ITEMS:
    p = R / rel
    md5 = hashlib.md5(p.read_bytes()).hexdigest()[:10] if p.exists() else "MISSING"
    out.append("| %s | `%s` | `%s` | %s |" % (tag, rel, md5, note))

out += ["", "## 刻意不在这趟（避免污染补丁 A 归因）", "",
        "- **S51/E8** `ops::silu`/`sigmoid` 极端负值归零修复（改所有 nvfp4 线性层数值）——下一趟，且需同时加极端负值回归用例。",
        "- **E9/S52** dflash2 可配置草稿宽度 K 的最小切片（改 dflash2 行为）——单独一轮。",
        "- **E7 回归测试** `E7_s50_regression_sketch.diff`（store 级页边界用例）——需要能编 tests 的窗口。",
        "- **E3 的 i8 平面步长 + S36 恢复**（256 下逐字节等价）、**E1 补丁 B**（`attention_valid` 契约）。", ""]

doc = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_landed_set.md")
doc.write_text("\n".join(out) + "\n", encoding="utf-8")
print("wrote %s (%d lines)" % (doc.name, len(out)))
print("\n".join(out[4:9]))
