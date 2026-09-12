#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 工作区里 vllm 相关目录 ==="
ls -d $J/*vllm* $J/*VLLM* $J/*1Cat* 2>/dev/null | head
echo
echo "=== dl/ 里 vllm 相关日志（按时间倒序） ==="
ls -lt --time-style=+%m-%d_%H:%M $J/dl/ 2>/dev/null | grep -iE 'vllm|1cat|accept' | head -12
echo
echo "=== 根目录的 vllm 启动脚本 / 配置 ==="
ls -lt --time-style=+%m-%d_%H:%M $J/*.sh $J/*.bat $J/*.py 2>/dev/null | grep -iE 'vllm|dflash|df2|1cat' | head -12
echo
echo "=== 1Cat-vLLM 目录结构（顶层 + 是否有 dflash2 支持） ==="
ls -1 $J/1Cat-vLLM 2>/dev/null | head -20
echo "--- 里面与 dflash2/dspark 有关的文件 ---"
ls -1 $J/1Cat-vLLM 2>/dev/null | grep -iE 'dflash|dspark|spec|README|run|serv' | head -10
