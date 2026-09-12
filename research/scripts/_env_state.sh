#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
PY="/mnt/c/vllm/venv-clean/Scripts/python.exe"
echo "=== 1) 当前装的是哪个版本 ==="
"$PY" -m pip show vllm 2>/dev/null | grep -E '^(Name|Version|Location)' 
echo
echo "=== 2) vllm 包里的编译扩展文件（单目录 ls 过滤） ==="
ls -1 /mnt/c/vllm/venv-clean/Lib/site-packages/vllm 2>/dev/null | grep -iE '_C|\.pyd|\.so|_stable' | head -10
echo "--- 老 venv 作对照（它当初能跑） ---"
ls -1 /mnt/c/vllm/venv/Lib/site-packages/vllm 2>/dev/null | grep -iE '_C|\.pyd|\.so|_stable' | head -10
echo
echo "=== 3) 回滚日志里 pip 的真实结论 ==="
grep -nE 'Successfully installed|Uninstalling|ERROR|error:' $J/dl/vllm_rollback.log 2>/dev/null | head -8
echo
echo "=== 4) fork 的 dflash2 方法名注册（单文件 55-70 行） ==="
sed -n '50,72p' $J/1Cat-vLLM/vllm/config/speculative.py 2>/dev/null
echo
echo "=== 5) 曾用于构建/获取 vllm 轮子的脚本（单目录，看名字） ==="
ls -1 $J/_z2_dl_wheelhouse.ps1 $J/supervise_build.ps1 $J/build_sm70.bat $J/run-vllm-fusion.bat 2>/dev/null
