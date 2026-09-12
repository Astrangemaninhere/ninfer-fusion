#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) _vllm_clean.py 60 行之后（请求 + 指标抓取） ==="
sed -n '60,120p' $J/_vllm_clean.py
echo
echo "=== 2) 之前参考跑的接受率（定点读单个文件） ==="
for L in dl/vref3.out dl/vref3.err dl/vllm_ref2.out dl/vllm_ref_run.log; do
  if [ -f "$J/$L" ]; then
    echo "--- $L ($(stat -c '%y' "$J/$L" | cut -c1-16), $(stat -c %s "$J/$L") B)"
    grep -iE 'accepted|drafted|accept|spec_decode|rate|position' "$J/$L" 2>/dev/null | tail -12
  fi
done
