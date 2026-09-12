#!/bin/bash
# GPU window A: pause training, run ONLY the e8 source dump (needs the current
# binary's NINFER_KVDUMP probe, not the pending rebuild), resume training.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== vram before ==="
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== pause training ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('kill ' + \$_.ProcessId); (Get-Process -Id \$_.ProcessId).Kill() }"
sleep 10
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== e8 source dump ==="
bash "$J/_e8_src_dump.sh" 100000
echo "=== resume training ==="
cd "$J" || exit 1
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat > dl\\train-resume-a.out 2>&1"
sleep 20
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo GPU_WINDOW_A_DONE
