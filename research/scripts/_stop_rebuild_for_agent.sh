#!/bin/bash
# 先停掉我自己的重建编排器（避免与代理的 make 抢），再派代理做修复循环
J=/mnt/c/Users/User/Documents/ziqinzhang
PAT=rebuild_after_s35
for p in $(pgrep -f "$PAT"); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in *"$PAT"*) echo "  TERM $p"; kill -TERM "$p" 2>/dev/null ;; esac
done
sleep 2
pgrep -f "$PAT" >/dev/null && echo "  仍有残留" || echo "  已停（把 make 让给修复代理）"
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in *ninfer-fusion*) echo "  TERM 在编进程 $p"; kill -TERM "$p" 2>/dev/null ;; esac
done
sleep 2
echo "=== 当前错误快照（给代理的起点） ==="
grep -E 'error:' /tmp/reb_ninfer_1.log 2>/dev/null | head -6 | cut -c1-160
