#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== 1) 实测日志里出现的 kv 相关字段 ==="
for f in /home/user/lmh2_df2_full.log /home/user/lmh_df2_full.log /home/user/lmh_dspark_full.log; do
  echo "--- $f ---"
  grep -oiE 'kv[_a-z]*[=: ]+[a-z0-9_]+' "$f" 2>/dev/null | sort -u | head -10
done
echo
echo "=== 2) request_log / engine_options 里 kv 字段打印点 ==="
grep -rn 'kv_dtype\|kv_quant\|"kv"' $R/src/serve/request_log.cpp 2>/dev/null | head -10
echo
echo "=== 3) CLI 默认 kv-dtype ==="
grep -rn -B3 -A6 'kv-dtype\|kv_dtype' $R/apps/cli/options.cpp 2>/dev/null | head -40
echo
echo "=== 4) 我实际用的命令（脚本原文） ==="
grep -n 'apps/ninfer' $J/_df2_align_run.sh $J/_lmhead_ab2.sh 2>/dev/null
