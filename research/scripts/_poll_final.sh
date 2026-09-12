#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "--- 链日志尾 ---"
tail -14 $J/dl/df2_final_chain.log 2>/dev/null
echo "--- runner stdout ---"
tail -12 $J/dl/vref_final_stdout.log 2>/dev/null
echo "--- runner 是否还在 ---"
if pgrep -f '/tmp/vd4.py' >/dev/null; then echo "runner 在跑"; else echo "runner 已结束"; fi
echo "--- vllm server 是否还在 ---"
if pgrep -f 'vllm.entrypoints.openai.api_server' >/dev/null; then echo "server 在跑"; else echo "server 已退出"; fi
echo "--- err 最后错误 ---"
grep -nE 'RuntimeError|Error:|error|Traceback' $J/dl/vref_df2c.err 2>/dev/null | tail -5
