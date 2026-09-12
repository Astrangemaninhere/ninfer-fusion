#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
F="$J/_collab/M_df2_serve_measure.md"
echo "=== 文件信息 ==="
ls -la --time-style=+%m-%d_%H:%M "$F" 2>/dev/null | cut -c25-95
echo
echo '=== 关键行：命令 / artifact / spec / 容量 / 50.9% ==='
grep -nE 'ninfer|serve|\.ninfer|spec|draft-tokens|max-context|kv-capacity|50\.9|4\.55|acceptance|tok/round|--greedy' "$F" 2>/dev/null | head -30 | cut -c1-150
echo
echo '=== 那次测量用的脚本（若文件里点名）==='
grep -oE '_[a-z0-9_]+\.(sh|bat)' "$F" 2>/dev/null | sort -u | head -6
echo
echo '=== 相关测量脚本是否存在 ==='
for s in _muse_serve_accept.sh _df2_serve_measure.sh _spec_4way.sh _window_k3.sh; do
  [ -f "$J/$s" ] && echo "  [在] $s  $(stat -c %y "$J/$s" | cut -c1-19)"
done
