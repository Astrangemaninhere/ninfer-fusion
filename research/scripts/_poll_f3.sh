#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== runner stdout（尾 20） ==="
tail -20 $J/dl/vref_f3_stdout.log 2>/dev/null
echo
echo "=== 有没有出数 ==="
grep -E 'ACCEPTED=|RATE=|PER_POS=|SpecDecoding|ready=' $J/dl/vref_f3_stdout.log 2>/dev/null | tail -8
echo
echo "=== err 里的错误 ==="
grep -nE 'RuntimeError|Error:|Traceback|FlashInfer|UVA' $J/dl/vref_f3.err 2>/dev/null | tail -5
echo
echo "=== 进程 ==="
if pgrep -f 'run_df2_ref.py' >/dev/null; then echo "runner 在跑"; else echo "runner 结束"; fi
if pgrep -f 'vllm.entrypoints' >/dev/null; then echo "server 在跑"; else echo "server 结束"; fi
echo "=== err 尾 6 ==="
tail -6 $J/dl/vref_f3.err 2>/dev/null
