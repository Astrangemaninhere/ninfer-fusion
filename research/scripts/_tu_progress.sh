#!/bin/bash
# 回答"还要多久"：去找 nvcc 的临时工作目录，数已产出的 cubin/sass，看每个内核平均耗时。
# 依据：nvcc 对每个 kernel 跑一次 cicc+ptxas，产物落在它的临时目录里；目录里有多少个
# 完成的内核产物，就是进度条。
echo "=== nvcc 与 ptxas ==="
for p in $(pgrep -x nvcc); do
  echo "  nvcc PID $p 已跑 $(ps -o etimes= -p $p)s  启动 $(ps -o lstart= -p $p | xargs)"
  echo "  其临时目录（/proc/$p/cwd 与 cmdline 里的 -o 推断）："
  tr '\0' '\n' < /proc/$p/cmdline | grep -E '^(-o|--output-directory)|tmp|nvcc' | head -5
done
ps -eo pid,etimes,time,rss,comm | grep ptxas | head -3

echo
echo "=== 找 nvcc 临时目录（最近 90 分钟内活动的）==="
for d in $(find /tmp -maxdepth 1 -name 'nvcc*' -newermt '-90 minutes' 2>/dev/null; \
           find /tmp -maxdepth 1 -type d -name 'tmp*' -newermt '-90 minutes' 2>/dev/null); do
  n_cubin=$(find "$d" -name '*.cubin' 2>/dev/null | wc -l)
  n_ptx=$(find "$d" -name '*.ptx' 2>/dev/null | wc -l)
  n_sass=$(find "$d" -name '*.sass' 2>/dev/null | wc -l)
  newest=$(find "$d" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1)
  echo "  $d  文件 $(find "$d" -type f 2>/dev/null | wc -l)  cubin=$n_cubin ptx=$n_ptx sass=$n_sass"
  [ -n "$newest" ] && echo "    最新文件: $(echo "$newest" | cut -d' ' -f2-) @ $(date -d @$(echo "$newest" | cut -d' ' -f1) +%H:%M:%S)"
done

echo
echo "=== 若找到内核产物：按 mtime 排出每个内核的耗时（= 进度与速率）==="
D=$(find /tmp -maxdepth 1 -name 'nvcc*' -newermt '-90 minutes' 2>/dev/null | head -1)
if [ -n "$D" ]; then
  find "$D" -name '*.cubin' -printf '%T@ %s %p\n' 2>/dev/null | sort -n | \
  awk -v d="$D" '{n++; if(p){printf "    #%02d  间隔 %6.1f s  大小 %8.2f MB  %s\n", n, $1-p, $2/1048576, $3} else {printf "    #%02d  (起点)  %8.2f MB  %s\n", n, $2/1048576, $3}; p=$1} END {print "    合计 " n " 个内核产物"}'
else
  echo "  （没找到 nvcc 临时目录：可能已清理，或用了 --keep 之外的模式）"
  echo "  退而求其次：看这个 TU 的 .o 是否已在写"
  ls -la --time-style=+%H:%M:%S /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o 2>/dev/null | cut -c25-110
fi
echo
echo "=== 内存/swap 现状 ==="
free -g | sed -n '2,3p'
