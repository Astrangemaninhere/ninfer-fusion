#!/bin/bash
# 调度占位：等 A5b+S52 批次结束 -> 跑 flag 探针 -> 批次2 的 nvcc 门禁会自动排在探针之后。
#
# 为什么这样排：四个批次首尾相接（构建→对照→批1→批2），探针要独占内存所以挤不进去。
# 但批2 的门禁是 "无编译器在跑"，只要探针的 nvcc 先起来，批2 就会自己等 —— 零改动即可插队。
# 每个 case 限时 25 分钟，避免一个 flag 组合把整条链拖住 2 小时。
set -u
export PATH="/home/user/.local/bin:$PATH"
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
B=$R/build
OUT=/home/user/jitprobe
LOG=$J/dl/jit_flags_probe.log
mkdir -p "$OUT"
exec >> "$LOG" 2>&1

echo "=================================================================="
echo "=== probe_arm 等待窗口 $(date '+%F %H:%M:%S') ==="
done_marker() { grep -q 'BATCH_BUILD_NEXT_DONE' "$J/dl/batch_build_next.log" 2>/dev/null; }
t0=$(date +%s)
while ! done_marker || pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; do
  if [ $(( $(date +%s) - t0 )) -gt 21600 ]; then echo "TIMEOUT 等窗口 (>6h)"; exit 2; fi
  sleep 30
done
echo "--- 窗口空了 $(date +%H:%M:%S)，开始探针 ---"
free -g | sed -n 2p

# 取真实编译命令（make -n 只打印不执行，安全）
OBJ=src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o
cd "$B" || exit 3
CMD=$(make -n "$OBJ" 2>/dev/null | grep -m1 -E 'nvcc' | sed -E 's#^ccache ##')
[ -z "$CMD" ] && { echo "取不到编译命令，放弃"; exit 4; }
echo "--- 基准命令（截断）---"
echo "$CMD" | cut -c1-200

CV=/usr/local/cuda-13.3/bin/nvcc
run_case() {  # $1=标签  $2=额外 flag（逗号分隔会被拆开）
  local tag=$1; shift
  local cmd; cmd=$(echo "$CMD" | sed -E "s#-o [^ ]*#-o $OUT/$tag.o#")
  echo "--- [$tag] 额外: ${*:-无} ---"
  timeout 1500 /usr/bin/time -v bash -c "NVCC_PREPEND_FLAGS='' $CV $* $cmd" 2>&1 \
    | grep -E 'Elapsed \(wall|Maximum resident' | sed 's/^/  /' || echo "  （超时 25 分钟，视为该 flag 不可用）"
  [ -f "$OUT/$tag.o" ] && ls -la "$OUT/$tag.o" | awk '{printf "   产物 %.1f MB\n", $5/1048576}'
  rm -f "$OUT/$tag.o"
}

run_case baseline
run_case ptxas_O2 -Xptxas -O2
# 去掉 -lineinfo（对 ptxas 工作量有直接影响）
CMD_NOL=$(echo "$CMD" | sed -E "s#-lineinfo##g" | sed -E "s#-o [^ ]*#-o $OUT/nolineinfo.o#")
echo "--- [nolineinfo] （从命令里删掉 -lineinfo）---"
timeout 1500 /usr/bin/time -v bash -c "NVCC_PREPEND_FLAGS='' $CV $CMD_NOL" 2>&1 \
  | grep -E 'Elapsed \(wall|Maximum resident' | sed 's/^/  /' || echo "  （超时）"
rm -f "$OUT/nolineinfo.o"

echo "=== 判读 ==="
echo "  * baseline 的时间/内存 = 现状；ptxas_O2 / nolineinfo 若明显下降，就是可用的减负手段"
echo "  * 并行上限 ≈ (MemAvailable - 3GB) / 该 flag 下的峰值内存（单 ptxas 实测尖峰 12.3 GB）"
echo "  * 任何减速推理的代价必须随后用 tok/s 与 token-id 等价性复核，不能只看编译快"
echo JIT_FLAGS_PROBE_DONE
echo PROBE_ARM_DONE
