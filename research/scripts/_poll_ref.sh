#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 内存 ==="
free -g | head -2
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -2
echo "=== 进程 ==="
if pgrep -f 'run_df2_ref.py' >/dev/null; then echo "runner 在跑"; else echo "runner 结束"; fi
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
echo "=== f3 stdout 尾 10 ==="
tail -10 $J/dl/vref_f3_stdout.log 2>/dev/null
echo "=== 是否出数 ==="
grep -E 'ACCEPTED=|RATE=|PER_POS=|ready=' $J/dl/vref_f3_stdout.log 2>/dev/null | tail -5
echo "=== f3 err 尾 6 ==="
tail -6 $J/dl/vref_f3.err 2>/dev/null
