#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) runner stdout 全文关键行 ==="
grep -E 'ready=|ACCEPTED=|DRAFTED=|RATE=|PER_POS=|SpecDecoding|text\[:60\]|finish=|request error|VREF_DF2_DONE' $J/dl/vref_f4_stdout.log 2>/dev/null | tail -20
echo
echo "=== 2) stdout 尾 12 行 ==="
tail -12 $J/dl/vref_f4_stdout.log 2>/dev/null
echo
echo "=== 3) server 日志里的 SpecDecoding 指标 ==="
grep -E 'SpecDecoding' $J/dl/vref_f4.out 2>/dev/null | tail -6
echo
echo "=== 4) 进程/GPU ==="
if pgrep -f 'run_df2_ref.py' >/dev/null; then echo "runner 仍在跑"; else echo "runner 已结束"; fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
echo "=== 5) err 尾 8 ==="
tail -8 $J/dl/vref_f4.err 2>/dev/null
