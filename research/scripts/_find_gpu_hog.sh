#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== GPU 上的 compute 进程（PID/显存） ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo
echo "=== 整体显存 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo
echo "=== WSL 侧还有哪些相关进程活着 ==="
ps -eo pid,stat,etime,cmd | grep -E 'ninfer|train_dflash|python3' | grep -v grep | cut -c1-120 | head -8
echo
echo "=== Windows 侧 python/训练进程 ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { '{0}  {1}' -f \$_.ProcessId, \$_.CommandLine.Substring(0,[Math]::Min(100,\$_.CommandLine.Length)) }" 2>/dev/null | tr -d '\r' | head -6
echo
echo "=== Muse 验收日志（新一轮） ==="
tail -12 "$J/dl/muse_accept_v2.log" 2>/dev/null | cut -c1-160
