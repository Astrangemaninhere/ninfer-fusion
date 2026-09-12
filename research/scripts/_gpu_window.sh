#!/bin/bash
# GPU window: pause the Windows training (precise PID match), verify the Muse
# row-scale fix, dump e8 sources for the §96 error analysis, then resume.
set -u
echo "=== pause training (match cmdline train_dflash2) ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('killing ' + \$_.ProcessId); Stop-Process -Id \$_.ProcessId -Force }"
sleep 8
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== muse recheck ==="
bash /mnt/c/Users/User/Documents/ziqinzhang/_muse_recheck.sh
echo "=== e8 source dump ==="
bash /mnt/c/Users/User/Documents/ziqinzhang/_e8_src_dump.sh
echo "=== resume training ==="
cd /mnt/c/Users/User/Documents/ziqinzhang
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat > dl\\train-retrain2.out 2>&1"
sleep 20
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo GPU_WINDOW_DONE
