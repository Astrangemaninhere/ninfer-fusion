#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## clean_vllm.log 里的接受率/投机指标 ##########"
grep -inE 'accept|spec|draft|acceptance|rate|Avg|throughput' $J/dl/clean_vllm.log 2>/dev/null | head -25
echo
echo "########## clean_vllm.log 开头的模型/参数 banner ##########"
head -25 $J/dl/clean_vllm.log 2>/dev/null
echo
echo "########## clean_vllm.err ##########"
cat $J/dl/clean_vllm.err 2>/dev/null | head -15
echo
echo "########## 启动脚本 ##########"
head -40 $J/_run_vllm_clean.bat 2>/dev/null
