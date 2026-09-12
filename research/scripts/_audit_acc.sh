#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 所有含 acceptance rate 的日志：文件 | 接受率 | 位置剖面 | 命令特征 ==="
for f in $J/dl/*.log $J/*.log; do
  [ -f "$f" ] || continue
  a=$(grep -m1 'dflash2 acceptance rate' "$f" 2>/dev/null | grep -oE '[0-9.]+%')
  [ -z "$a" ] && continue
  p=$(grep -m1 'dflash2 accepted by pos' "$f" 2>/dev/null | sed 's/.*pos *//')
  sp=$(grep -m1 'decode speed' "$f" 2>/dev/null | grep -oE '[0-9.]+ tok/s')
  n=$(grep -m1 -oE 'max-new [0-9]+' "$f" 2>/dev/null)
  dt=$(grep -m1 -oE 'draft-tokens [0-9]+' "$f" 2>/dev/null)
  kv=$(grep -m1 -oE 'kv-dtype [a-z0-9]+' "$f" 2>/dev/null)
  printf "%-42s %-8s %-20s %-12s %-10s %-12s\n" "$(basename $f)" "$a" "$p" "${sp:-?}" "${n:-?}" "${kv:-?}"
done
echo
echo "=== 4.81% 出现在哪些文件 ==="
grep -rl '4\.81%' $J/dl/*.log $J/*.log 2>/dev/null
echo
echo "=== masktok A/B 两臂的关键行 ==="
for f in $J/dl/masktok_*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename $f) ---"
  grep -E 'acceptance rate|accepted by pos|decode speed|draft window|rounds' "$f" | head -6
done
