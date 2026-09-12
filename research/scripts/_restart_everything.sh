#!/bin/bash
# 重建（两次地雷修好后）+ 正确起 K4（K4 用脚本文件起，避免 $J 引号坑）
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 "$J/_fix_scanner.py"
# 等一下上一轮 rebuild 编排器（它在重试 ninfer-serve，大概率都失败）自然结束
for i in $(seq 1 30); do
  pgrep -f '_rebuild_after_s35.sh' >/dev/null || break
  sleep 10
done
pkill -f '_rebuild_after_s35.sh' 2>/dev/null || true
sleep 2
echo "=== 上次 rebuild 结论 ==="; tail -6 "$J/dl/rebuild_after_s35.log" | cut -c1-140
echo
echo "=== 重新起重建 ==="
setsid nohup bash "$J/_rebuild_after_s35.sh" > /dev/null 2>&1 < /dev/null &
sleep 20
tail -4 "$J/dl/rebuild_after_s35.log" | cut -c1-140
echo
echo "=== 起 K4（这次用文件，不走 -c 引号） ==="
setsid nohup bash "$J/_window_k4.sh" > /dev/null 2>&1 < /dev/null &
sleep 6
pgrep -af '_window_k4.sh' | cut -c1-70 || echo "  K4 未起"
tail -3 "$J/dl/window_k4.log" 2>/dev/null | cut -c1-120
