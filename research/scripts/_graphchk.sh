#!/bin/bash
# 判定当前二进制是否走图：显式 ceiling 16 vs 0，k=1/3
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
M=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
OUT=$J/_graphchk.txt
{
  echo "=== 判定当前二进制是否走图  起 $(date '+%H:%M:%S') ==="
  echo "源码里的默认 ceiling："
  grep -n 'graph_capture_ceiling = ' /home/user/ninfer-fusion/apps/cli/options.h \
    /home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/layouts.h \
    /home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/program.h 2>/dev/null | sed 's/^/  /'
  cd /home/user/ninfer-fusion/build || exit 3
  pkill -9 -x ninfer 2>/dev/null; sleep 2
  printf "  %-22s %-9s %s\n" 配置 tok/s 说明
  for cc in 16 0; do
    for k in 1 3; do
      timeout 600 ./apps/ninfer "$M" --prompt "$P" --max-new 64 --max-context 4096 \
        --no-thinking --greedy --spec mtp --draft-tokens "$k" --lm-head-draft \
        --graph-capture-ceiling "$cc" > "$J/gc_${cc}_$k.log" 2>&1
      sp=$(grep -m1 'decode speed' "$J/gc_${cc}_$k.log" | grep -oE '[0-9.]+')
      al=$(grep -m1 'mtp acceptance length' "$J/gc_${cc}_$k.log" | grep -oE '[0-9.]+')
      printf "  %-22s %-9s AL=%s\n" "ceiling=$cc k=$k" "${sp:-?}" "${al:-?}"
    done
  done
  echo "宿主: $(uptime | sed 's/.*load average/load/')"
  echo "=== 完成 $(date '+%H:%M:%S') ==="
} > "$OUT" 2>&1
echo written
