#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 停训练（Stop-Process -Force 有效；.Kill() 今天已证明无效） ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  killing ' + \$_.ProcessId; Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" 2>/dev/null | tr -d '\r'
sleep 5
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  still alive ' + \$_.ProcessId; Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null | tr -d '\r'
sleep 8
echo "  剩余训练进程: $(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' }).Count" 2>/dev/null | tr -d '\r')"
echo "  显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo
echo "=== 2) 停掉还在跑的编译（避免编译结果与源码不一致；改完 bug 一次编） ==="
for p in $(pgrep -f 'rebuild_after_s35|make|bin/nvcc|cc1plus' 2>/dev/null); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in
    *rebuild_after_s35*|*"make ninfer"*|*nvcc*|*cc1plus*) echo "  TERM $p $(echo "$cmd" | cut -c1-70)"; kill -TERM "$p" 2>/dev/null ;;
  esac
done
sleep 4
pgrep -f 'bin/nvcc|cc1plus' >/dev/null && echo "  仍有编译进程" || echo "  编译已停"
echo
echo "=== 3) 基线：训练进度与最近 checkpoint（用户说训练先放下） ==="
tail -2 "$J/dl/train-dflash2.log" | cut -c1-130
ls -t "$J/data/dflash2_ckpts/"*.pt 2>/dev/null | head -2 | sed 's/^/  ckpt: /'
