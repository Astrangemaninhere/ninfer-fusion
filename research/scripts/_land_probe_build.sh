#!/bin/bash
# 落 T3 的 dflash2 位置/草稿探针（env-gated，零行为变化），然后 -j8 重编。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
cd "$R" || exit 1

echo '=== 备查：补丁存在性与 dry-run ==='
P=$C/build/T3_df2_position_probe.diff
ls -la "$P" 2>/dev/null | cut -c25-100
if patch -p1 --dry-run < "$P" >/dev/null 2>&1; then
  patch -p1 -b < "$P" && echo "  applied(raw)"
else
  tr -d '\r' < "$P" > /tmp/probe.diff
  patch -p1 -b --dry-run < /tmp/probe.diff >/dev/null 2>&1 && patch -p1 -b < /tmp/probe.diff && echo "  applied(cr)"
fi

echo '=== 落点确认 ==='
grep -n 'NINFER_DF2DBG' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" 2>/dev/null | head -3 | cut -c1-130

echo '=== touch + -j8 重编 ==='
touch "$R/src/targets/qwen3_6/impl/runtime/program_impl.h"
cd "$R/build" || exit 2
PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 setsid nohup bash "$J/_par_build.sh" ninfer ninfer-serve >/dev/null 2>&1 &
sleep 40
echo "  nvcc 并行: $(pgrep -c -x nvcc 2>/dev/null || echo 0)"
grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1
free -g | sed -n 2p
date +%H:%M:%S
