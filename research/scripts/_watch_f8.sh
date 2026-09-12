#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
L=$J/dl/run_f8.log
for i in $(seq 1 220); do
  grep -qE 'F8_DONE|未就绪' "$L" 2>/dev/null && break
  sleep 5
done
echo "=== F8 结束 $(date '+%H:%M:%S') ==="
tail -20 "$L"
echo "=== 关键数字 ==="
grep -E 'ACCEPTANCE=|PER_POS=|text\[:70\]|完成 ' "$L" 2>/dev/null | tail -8
echo "=== srv SpecDecoding ==="
grep -E 'SpecDecoding' $J/dl/srv_f8.out 2>/dev/null | tail -4
echo WATCH_F8_DONE
