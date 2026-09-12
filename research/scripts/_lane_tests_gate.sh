#!/bin/bash
# 门后待命：等 ninfer 编译彻底结束 + 内存回血 => 编并跑那 5 个 lane 测试（rope_delta 假设的可证伪门）。
# 现在只挂不等资源，不占内存。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
LOG=$J/dl/lane_tests_gate.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== lane 测试门后待命 $(date '+%F %H:%M:%S') ==="

compiler_alive() { pgrep -x nvcc >/dev/null 2>&1; }
orchestrator_alive() { pgrep -f '_par_build[.]sh' >/dev/null 2>&1; }
mem_ok() { [ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -gt 8388608 ]; }

t0=$(date +%s)
while compiler_alive || orchestrator_alive || ! mem_ok; do
  if [ $(( $(date +%s) - t0 )) -gt 21600 ]; then echo "TIMEOUT (>6h)"; exit 2; fi
  sleep 30
done
echo "--- 门开 $(date +%H:%M:%S)（无编译、内存 >8GB）---"
free -g | sed -n '2,3p'

cd "$R/build" || exit 3
TARGETS="ninfer_rope_test ninfer_argmax_test ninfer_position_test ninfer_prepare_masked_block_test ninfer_sigmoid_mul_test"
echo "--- 编译测试目标 ---"
make -j4 $TARGETS 2>&1 | tail -8
echo "  make rc=${PIPESTATUS[0]}"

echo "--- 逐个跑（含 A3 的 sigmoid 用例）---"
for t in $TARGETS; do
  b="$R/build/tests/$t"
  [ -x "$b" ] || { echo "  [缺] $t"; continue; }
  out=$(timeout 600 "$b" 2>&1); rc=$?
  echo "  [$t] rc=$rc  $(echo "$out" | grep -viE '^\s*$' | tail -2 | tr '\n' ' ' | cut -c1-160)"
  echo "$out" > /home/user/lt_$t.log
done

echo "--- 判读提示 ---"
echo "  * rope/position 两个测试是 'dflash2 verify 漏 rope_delta、partial RoPE 逐 lane 起点' 假设的可证伪门："
echo "    若它们 FAIL => 与上游契约确实不一致（假设得到支持）；若全 PASS => 该假设排除，需另找。"
echo "=== done $(date '+%F %H:%M:%S') ==="
echo LANE_TESTS_GATE_DONE
