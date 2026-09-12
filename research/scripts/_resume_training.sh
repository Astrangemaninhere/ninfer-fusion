#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 恢复训练（从 step_001900 续；实验已完成，GPU 该还回去） ==="
if [ -f "$J/_train_df2_resume.bat" ]; then
  powershell.exe -NoProfile -Command "Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','C:\Users\User\Documents\ziqinzhang\_train_df2_resume.bat' -WindowStyle Hidden" 2>&1 | tr -d '\r'
  sleep 45
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  training resumed pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r'
  tail -2 "$J/dl/train-dflash2.log" | cut -c1-130
else
  echo "  resume bat 缺失"
fi
echo
echo "=== 2) 显存（训练应重新占上） ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo
echo "=== 3) 编译进度（CPU 侧继续） ==="
tail -2 /tmp/reb_ninfer_1.log 2>/dev/null | cut -c1-120
