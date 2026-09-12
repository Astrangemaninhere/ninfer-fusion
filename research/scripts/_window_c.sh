#!/bin/bash
# GPU window C: pause the training (frees ~10GB host so the in-flight ptxas
# stops thrashing), wait for the build, force-rebuild the TUs whose headers
# changed after their compile started, run the Muse + e8 verifications, resume.
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== pause training ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('kill ' + \$_.ProcessId); (Get-Process -Id \$_.ProcessId).Kill() }"
sleep 12
"/mnt/c/Windows/System32/nvidia-smi.exe" --query-gpu=memory.used,memory.free --format=csv,noheader
echo "=== wait for the in-flight build ==="
t0=$(date +%s)
while pgrep -x make >/dev/null; do sleep 10; done
echo "build done after $(( $(date +%s) - t0 ))s"
echo "=== force-rebuild the TUs whose headers changed mid-build ==="
cd /home/user/ninfer-fusion/build || exit 1
touch /home/user/ninfer-fusion/src/ops/kernel/gqa_attention_decode_i8.cuh \
      /home/user/ninfer-fusion/src/ops/kernel/gqa_attention_prefill_i8.cuh \
      /home/user/ninfer-fusion/src/ops/kernel/gqa_isoquant_row_scale.cuh \
      /home/user/ninfer-fusion/src/ops/launcher/gqa_attention_decode.cu \
      /home/user/ninfer-fusion/src/ops/wrapper/gqa_attention.cpp
make ninfer -j4 2>&1 | tail -4
ls -l --time-style=+%H:%M:%S /home/user/ninfer-fusion/build/apps/ninfer
echo "=== muse verification ==="
bash "$J/_muse_verify.sh"
echo "=== e8 post-fix probe ==="
bash "$J/_e8_postfix.sh"
echo "=== resume training ==="
cd "$J" || exit 1
cmd.exe /c "start \"\" /b cmd /c _train_df2_resume.bat"
sleep 25
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { Write-Output ('resumed pid ' + \$_.ProcessId) }"
echo WINDOW_C_DONE
