#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== K4 进度（Muse 验收 / 是否已续训） ==="
tail -12 "$J/dl/window_k4.log" | cut -c1-160
echo
echo "=== GPU 占用 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -4
echo
echo "=== 是否已有训练进程 ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  training pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r' || true
echo
echo "=== Muse 验收结果 ==="
grep -E 'MUSE|PASS|FAIL|REFUSED' "$J/dl/window_k4.log" 2>/dev/null | tail -8 | cut -c1-150
