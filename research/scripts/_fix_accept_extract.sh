#!/bin/bash
# 用正确字段（summary <backend> acceptance rate）重新提取所有对比日志里的接受率，
# 顺便证明旧提取器（spec_accept_rate=）确实取不到值。
echo '=== 正确字段提取（cmp_*.log / r1r2_*.log / b2_*.log / width_*.log）==='
for f in /home/user/cmp_*.log /home/user/r1r2_*.log /home/user/b2_*.log /home/user/width_*.log /home/user/ab2_*.log; do
  [ -f "$f" ] || continue
  line=$(grep -oE 'summary +[a-z0-9]* ?acceptance rate +[0-9.]+%?' "$f" 2>/dev/null | head -1)
  [ -z "$line" ] && line=$(grep -iE 'acceptance rate' "$f" 2>/dev/null | head -1 | cut -c1-90)
  pos=$(grep -oE 'accepted by pos +[0-9,]+' "$f" | tail -1 | sed 's/.*pos *//')
  spd=$(grep -oE 'decode speed +[0-9.]+' "$f" | head -1 | grep -oE '[0-9.]+')
  printf '  %-24s %-52s pos=[%s] decode=%s\n' "$(basename "$f" .log)" "${line:-（无该字段）}" "${pos:-}" "${spd:-}"
done | head -30
echo
echo '=== 反证：旧提取器在同一批日志上取到什么 ==='
n=0
for f in /home/user/cmp_*.log; do
  v=$(grep -oE 'spec_accept_rate=[0-9.]+' "$f" | head -1)
  [ -n "$v" ] && n=$((n+1))
done
echo "  spec_accept_rate= 在 $(ls /home/user/cmp_*.log 2>/dev/null | wc -l) 个日志里命中 $n 次（0 即为采集 bug）"
