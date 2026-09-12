#!/bin/bash
# 按 PID 重启重建编排器：模式只出现在本文件内容里，不出现在 shell cmdline 上
# （上一条命令就是因为把模式写进了 -c 的 cmdline，pkill -f 把自己杀了）。
J=/mnt/c/Users/User/Documents/ziqinzhang
PAT=rebuild_after_s35
for p in $(pgrep -f "$PAT"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *"$PAT"*) echo "  TERM $p ($(echo "$cmd" | cut -c1-60))"; kill -TERM "$p" 2>/dev/null ;;
  esac
done
sleep 3
pgrep -f "$PAT" >/dev/null && echo "  仍有残留" || echo "  旧实例已清"
setsid nohup bash "$J/_rebuild_after_s35.sh" > /dev/null 2>&1 < /dev/null &
sleep 25
echo "=== 新的重建进度 ==="
tail -5 "$J/dl/rebuild_after_s35.log" | cut -c1-150
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do
  echo "  在编: $(ps -o etime= -p $p | tr -d ' ') $(tr '\0' ' ' < /proc/$p/cmdline | grep -oE '[^ ]+\.(cc|cpp|cu)' | tail -1)"
  break
done
