#!/bin/bash
# ptxas 开销的 A/B 探针：在 /tmp 里单独编那个 40 分钟的大 TU，只改编译 flag，
# 量 ① 墙钟时间 ② 峰值内存。**必须先知道这两个数**，才能决定：
#   * 要不要给 ptxas 降优化等级 / 去掉 -lineinfo；
#   * 拆完 TU 之后能同时跑几个（拆了但只能跑一个 = 白拆）。
# 为什么必须用独立 -o 和 /tmp：绝不能碰正在构建的 build 树，也不能和它抢对象。
set -u
export PATH="/home/user/.local/bin:$PATH"
R=/home/user/ninfer-fusion
B=$R/build
OUT=/home/user/jitprobe
mkdir -p "$OUT"
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/jit_flags_probe.log
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== ptxas flag 探针 $(date '+%F %H:%M:%S') ==="

# 从 build.make 里取出真实的编译命令（含全部 -I/-D），再替换输入输出与 flag。
OBJ=src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o
cd "$B" || exit 2
CMD=$(make -n "$OBJ" 2>/dev/null | grep -m1 -E 'nvcc|ccache' )
if [ -z "$CMD" ]; then echo "取不到编译命令，放弃"; exit 3; fi
echo "--- 原始命令（截断）---"
echo "$CMD" | cut -c1-220
echo

run_case() {  # $1=标签 $2..=额外 flag
  local tag=$1; shift
  local extra=("$@")
  local cmd=$CMD
  # 输出改到 /tmp，避免碰 build 树
  cmd=$(echo "$cmd" | sed -E "s#-o [^ ]*#-o $OUT/$tag.o#")
  cmd=$(echo "$cmd" | sed -E "s#^ccache ##")
  echo "--- [$tag] 额外: ${extra[*]:-无} ---"
  # shellcheck disable=SC2086
  /usr/bin/time -v bash -c "NVCC_PREPEND_FLAGS='' /usr/local/cuda-13.3/bin/nvcc ${extra[*]} $cmd" \
      2>&1 | grep -E 'Elapsed \(wall|Maximum resident' | sed 's/^/  /'
  ls -la "$OUT/$tag.o" 2>/dev/null | awk '{printf "   产物 %.1f MB\n", $5/1048576}'
  echo
}

run_case baseline
run_case ptxas_O2      -Xptxas -O2
run_case ptxas_O1      -Xptxas -O1
run_case nolineinfo_none ""
echo '=== 说明 ==='
echo '  * baseline 应与 build 树里的对象逐字节同源（同 flag 同输入）'
echo '  * ptxas_O2/O1 若时间大幅下降且峰值内存也降 -> 值得全局启用，但必须再验推理速度'
echo '  * -lineinfo 用下一行单独测（需要从命令里删掉，用 sed）'
# -lineinfo 单独：从命令里剔除
cmd_nol=$(echo "$CMD" | sed -E "s#-lineinfo##g" | sed -E "s#-o [^ ]*#-o $OUT/nolineinfo.o#")
echo "--- [nolineinfo] （从命令中删除 -lineinfo）---"
/usr/bin/time -v bash -c "NVCC_PREPEND_FLAGS='' /usr/local/cuda-13.3/bin/nvcc $cmd_nol" \
    2>&1 | grep -E 'Elapsed \(wall|Maximum resident' | sed 's/^/  /'
echo JIT_FLAGS_PROBE_DONE
