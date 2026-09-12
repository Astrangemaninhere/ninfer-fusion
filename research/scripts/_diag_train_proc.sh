#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 训练日志是否还在增长（判断活着 vs 僵死） ==="
a=$(stat -c %s "$J/dl/train-dflash2.log"); echo "  size now: $a"
sleep 20
b=$(stat -c %s "$J/dl/train-dflash2.log"); echo "  size +20s: $b"
if [ "$b" -gt "$a" ]; then echo "  ⇒ 日志在增长：训练仍在跑（Kill 未生效，或它自行恢复了）"; else echo "  ⇒ 日志停滞：进程僵死"; fi
tail -2 "$J/dl/train-dflash2.log" | cut -c1-140
echo
echo "=== 进程状态与 CPU 占用（僵死的 CUDA 进程通常 CPU=0 且状态挂起） ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter 'ProcessId=16344' | ForEach-Object { '  pid=' + \$_.ProcessId + ' 状态=' + \$_.ExecutionState + ' 线程=' + \$_.ThreadCount + ' CPU时间=' + \$_.KernelModeTime }" 2>/dev/null | tr -d '\r'
for p in $(pgrep -f 'train_dflash' 2>/dev/null); do echo "  WSL 侧可见 train pid: $p $(cat /proc/$p/stat 2>/dev/null | awk '{print $3}')"; done
echo
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
