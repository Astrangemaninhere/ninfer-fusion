#!/bin/bash
# 正确策略：两个巨兽 TU 先 -j1 串行编完，再 -j8 编剩下的（避免 2×12 GB 撞内存）。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
LOG=$J/dl/build_serial_monsters.log
exec >> "$LOG" 2>&1
echo "=============================================="
echo "=== 巨兽串行 + 其余并行 $(date '+%F %H:%M:%S') ==="
free -g | sed -n 2p

# 确认没有别的编译在跑（避免再次叠加）
while pgrep -x nvcc >/dev/null 2>&1; do echo "  等已有 nvcc 退出…"; sleep 10; done

cd "$R/build" || exit 4001
MONSTERS="src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o
src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_e8.cu.o
src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_g35.cu.o
src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_muse.cu.o
src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_prefill.cu.o"

echo "--- 第 1 阶段：巨兽逐个 -j1（每个都独占内存） ---"
for m in $MONSTERS; do
  if [ -f "$m" ] && [ "$m" -nt "$(echo "$m" | sed 's#\.o$##; s#src/CMakeFiles/ninfer_ops.dir/##')" ]; then
    echo "  [已新] $m"; continue
  fi
  echo "  --- make -j1 $m  $(date +%H:%M:%S) ---"
  free -g | sed -n 2p
  timeout 3600 make -j1 "$m" 2>&1 | tail -4
  echo "    rc=${PIPESTATUS[0]}"
done

echo "--- 第 2 阶段：其余全部 -j8 ---"
echo "  $(date +%H:%M:%S)"
make ninfer ninfer-serve -j8 2>&1 | tail -12
echo "  make rc=${PIPESTATUS[0]}"

echo "--- 产物 ---"
ls -l --time-style=+%m-%d_%H:%M "$R/build/apps/ninfer" "$R/build/apps/ninfer-serve" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
echo "=== 结束 $(date '+%F %H:%M:%S') ==="
echo BUILD_SERIAL_MONSTERS_DONE
