#!/bin/bash
echo "=== .wslconfig（单文件） ==="
F="/mnt/c/Users/User/.wslconfig"
if [ -f "$F" ]; then cat "$F"; else echo "(没有 .wslconfig)"; fi
echo
echo "=== 宿主内存（Windows 侧） ==="
/mnt/c/Windows/System32/wbem/WMIC.exe ComputerSystem get TotalPhysicalMemory 2>/dev/null | tr -d '\r' | head -3
echo
echo "=== WSL 可见内存 ==="
free -g | head -2
echo
echo "=== 上次运行的峰值线索：vllm 日志里的显存/内存相关行 ==="
tail -25 /mnt/c/Users/User/Documents/ziqinzhang/dl/vref_f3.err 2>/dev/null | head -12
