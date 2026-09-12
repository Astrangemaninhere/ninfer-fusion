#!/bin/bash
# GPU window B: pause the Windows training, verify the Muse row-scale fix
# (no-NaN + 1..6 -> 7 + Chinese sample), run the e8 pre/post-rotation quality
# probe, then resume the training from its newest checkpoint.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== vram before ==="
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== pause training ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('kill ' + \$_.ProcessId); (Get-Process -Id \$_.ProcessId).Kill() }"
sleep 12
echo "=== vram after kill (expect a large free figure) ==="
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== muse verification ==="
bash "$J/_muse_verify.sh"
echo "=== e8 post-fix probe ==="
bash "$J/_e8_postfix.sh"
echo "=== resume training ==="
cd "$J" || exit 1
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 25
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo WINDOW_B_DONE
