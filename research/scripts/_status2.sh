#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== ninfer 重编（A5b 回退版）$(date +%H:%M:%S) ==="
echo "  nvcc=$(pgrep -c -x nvcc 2>/dev/null || echo 0)  编排=$(pgrep -c -f '_par_build[.]sh' 2>/dev/null || echo 0)"
echo "  进度=$(grep -oE '\[[ 0-9]+%\]' "$J/dl/par_build.log" 2>/dev/null | tail -n 1)"
tail -n 2 "$J/dl/par_build.log" 2>/dev/null | cut -c1-105
echo "  二进制: $(stat -c '%y %s' /home/user/ninfer-fusion/build/apps/ninfer 2>/dev/null | cut -c1-32)"
echo
echo "=== A5b 回退是否真的在树里 ==="
grep -n 'set_i32_scalar(attention_valid' /home/user/ninfer-fusion/src/targets/qwen3_6/impl/runtime/dflash_impl.h | cut -c1-120
echo
echo "=== 干净 vLLM（端口 8127）==="
tail -n 5 "$J/dl/clean_run.log" 2>/dev/null | cut -c1-120
