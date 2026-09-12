#!/bin/bash
# 修掉树里那个"声明写在函数体内"的构建拦路石（第三次同类 bug），然后重启 -j8 构建。
# 依据：loader.h:154 的声明落在 kv_rowscale_sidecar_check 的函数体里 => 只在块作用域可见
# => decoder_state.cpp:172 在命名空间作用域调用就报 "is not a member"。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
C=$J/_collab
export PATH="/home/user/.local/bin:$PATH"

echo '=== ① 备份 + dry-run N3 的声明归位补丁 ==='
cp -f "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" "/home/user/loader.h.bak_before_declfix"
md5sum "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | sed 's/^/  修前 md5 /'
P="$C/N3_loader_declfix.diff"
if [ -f "$P" ]; then
  tr -d '\r' < "$P" > /tmp/declfix.diff
  patch -p1 --dry-run -d "$R" < /tmp/declfix.diff && echo "  dry-run OK" || echo "  dry-run 失败（可能已含别处改动）"
else
  echo "  缺补丁 $P —— 改为手工归位"
fi

echo
echo '=== ② 应用（dry-run 通过才落）==='
if patch -p1 --dry-run -d "$R" < /tmp/declfix.diff >/dev/null 2>&1; then
  patch -p1 -b -d "$R" < /tmp/declfix.diff && echo "  已应用"
else
  echo "  改为手工：把 150-157 那段（注释+声明）移出函数体，放到 namespace 作用域'
  python3 - <<'PY'
import pathlib, re
p = pathlib.Path("/home/user/ninfer-fusion/src/ops/kernel/gqa_isoquant_row_scale_loader.h")
s = p.read_text(encoding="utf-8", errors="surrogateescape")
decl = """// N3/S28 round 1 engine hook (definition: gqa_isoquant_row_scale_loader.cu).
// Reads NINFER_KV_ROWSCALE; unset/empty => false (feature off). Parses and validates
// against the loaded model identity, then uploads the table to the device constant.
// Any validation failure THROWS (no silent fallback to the baked table).
bool kv_rowscale_sidecar_apply_from_env(std::uint32_t model_layers,
                                        std::uint32_t model_kv_heads,
                                        std::uint32_t model_head_dim,
                                        std::uint64_t model_hash);
"""
if s.count(decl) == 1:
    s2 = s.replace(decl, "")
    anchor = "namespace ninfer::ops {\n"
    if s2.count(anchor) == 1:
        s2 = s2.replace(anchor, anchor + "\n" + decl, 1)
        p.write_text(s2, encoding="utf-8", errors="surrogateescape")
        print("  手工归位完成：声明已移到 namespace 作用域")
    else:
        print("  找不到 namespace 锚点，未改动")
else:
    print("  声明文本匹配 %d 次，未改动（需人工）" % s.count(decl))
PY
fi

echo
echo '=== ③ 校验：声明是否已在 namespace 作用域（缩进为 0 且不在函数体内）==='
grep -n 'kv_rowscale_sidecar_apply_from_env' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cut -c1-110
awk '/^bool kv_rowscale_sidecar_apply_from_env/{print "  ✓ 顶格声明（namespace 作用域）行 " NR}' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h"

echo
echo '=== ④ 让 decoder_state.cpp 重编（touch 它），然后重启 -j8 构建 ==='
touch "$R/src/targets/qwen3_6/impl/state/decoder_state.cpp"
pgrep -x nvcc >/dev/null 2>&1 && { echo "  还有 nvcc 在跑，等它停"; while pgrep -x nvcc >/dev/null 2>&1; do sleep 10; done; }
nohup setsid bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 20
echo "  nvcc 并行数: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
tail -3 "$J/dl/par_build.log" | cut -c1-120
date +%H:%M:%S
