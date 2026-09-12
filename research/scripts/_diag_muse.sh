#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== Muse 验收脚本用了哪些日志 ==="
grep -nE 'log=|BIN|--spec|PORT' "$J/_muse_serve_accept.sh" 2>/dev/null | head -12 | cut -c1-150
echo
echo "=== 最近的 muse serve 日志 ==="
ls -lt --time-style=+%H:%M /home/user/muse*.log 2>/dev/null | head -5 | awk '{print "  ", $6, $5, $NF}'
for f in $(ls -t /home/user/muse*.log 2>/dev/null | head -2); do
  echo "--- $f (tail) ---"
  tail -12 "$f" | cut -c1-150
done
echo
echo "=== K4 第 5 步的完整输出片段 ==="
sed -n '/5\/5/,$p' "$J/dl/window_k4.log" 2>/dev/null | head -20 | cut -c1-150
echo
echo "=== 当前 serve 进程（GPU 上那个） ==="
pgrep -af 'ninfer-serve' | cut -c1-130
