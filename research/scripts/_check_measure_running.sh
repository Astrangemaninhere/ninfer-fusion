#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 测量 watcher 状态 ==="
tail -6 "$J/dl/postbuild_measure.log" 2>/dev/null | cut -c1-150
echo
echo "=== 测量是否在跑（serve 进程 / 断言） ==="
pgrep -af 'ninfer-serve' | cut -c1-110 || echo "  当前没有 serve 进程"
pgrep -af '_post_build_measure.sh' | cut -c1-70 || echo "  watcher 不在了（已结束？）"
echo
echo "=== 结果文件 ==="
ls -l --time-style=+%H:%M "$J/_collab/M_patchA_effect.md" 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
cat /home/user/pb_verdict.txt 2>/dev/null | head -12
cat /tmp/pb_rows 2>/dev/null | sed 's/^/  /'
echo
echo "=== K4 状态 ==="
tail -5 "$J/dl/window_k4.log" 2>/dev/null | cut -c1-140
echo
echo "=== 顺便：18:00 汇报有没有产出 ==="
ls -l --time-style=+%H:%M "$J/_report_1800.md" 2>/dev/null | awk '{print "  ", $6, $5, $NF}' || echo "  （还没有）"
