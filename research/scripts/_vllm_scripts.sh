#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "########## _vllm_clean.py ##########"
cat $J/_vllm_clean.py 2>/dev/null | head -60
echo
echo "########## _vllm_clean_prob.py（若有差异） ##########"
grep -nE 'model|spec|dflash|dspark|draft|num_spec|prompt|tensor_parallel|quant' $J/_vllm_clean_prob.py 2>/dev/null | head -20
echo
echo "########## vllm_fg2.log（09:36，80KB，之前的运行）接受率 ##########"
grep -inE 'accept|spec_decode|draft|Accepted|metric' $J/dl/vllm_fg2.log 2>/dev/null | head -20
