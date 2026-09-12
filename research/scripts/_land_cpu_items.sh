#!/bin/bash
# CPU 落地：先 dry-run 门禁，再落 UP1 的两份补丁；然后判清测试的性质与当前构建状态。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
C=$J/_collab
R=/home/user/ninfer-fusion
export PATH="/home/user/.local/bin:$PATH"
cd "$R" || exit 1

apply() {  # $1=补丁
  local f=$1
  [ -f "$f" ] || { echo "  缺 $f"; return 1; }
  if patch -p1 -b --dry-run < "$f" >/dev/null 2>&1; then patch -p1 -b < "$f" && echo "  applied(raw): $(basename "$f")"; return 0; fi
  local t=/tmp/up1_$(basename "$f"); tr -d '\r' < "$f" > "$t"
  if patch -p1 -b --dry-run < "$t" >/dev/null 2>&1; then patch -p1 -b < "$t" && echo "  applied(cr): $(basename "$f")"; return 0; fi
  echo "  打不上: $(basename "$f")"; return 1
}

echo '=== ① 落 lane 测试 ==='
apply "$C/UP1_df2_lane_tests.diff"
echo '=== ② 落 ignore_eos ==='
apply "$C/UP1_ignore_eos.diff"

echo
echo '=== ③ 这 5 个测试是什么性质（CPU 还是 CUDA）==='
for f in $(grep -E '^\+\+\+ ' "$C/UP1_df2_lane_tests.diff" | sed 's|^+++ b/||' | sort -u); do
  [ -f "$R/$f" ] || continue
  cuda=$(grep -cE '#include <cuda|cuda_runtime|__global__|<<<|Tensor' "$R/$f" 2>/dev/null || echo 0)
  gtest=$(grep -cE 'gtest|TEST\(|EXPECT_' "$R/$f" 2>/dev/null || echo 0)
  printf '  %-58s CUDA线索=%-4s gtest=%s\n' "$f" "$cuda" "$gtest"
done
echo '  （CUDA线索=0 则纯主机可编可跑，能和 ninfer 编译并行）'

echo
echo '=== ④ 当前构建状态（决定测试编译能不能插进去）==='
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  编排=$(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)  进度=$(grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -1)"
echo "  二进制: $(stat -c '%y' $R/build/apps/ninfer 2>/dev/null | cut -c1-19)  内存可用: $(awk '/MemAvailable/{printf "%d MB", $2/1024}' /proc/meminfo)"
echo "  ignore_eos 是否已在树: $(grep -rc 'ignore_eos' $R/apps/ 2>/dev/null | grep -v ':0' | head -3 | tr '\n' ' ')"
