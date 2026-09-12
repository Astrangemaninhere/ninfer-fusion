#!/bin/bash
# 暂停续训（每 100 步有 checkpoint，代价小）→ 腾出 GPU 跑"verify vs plain"判别实验
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 记录当前训练进度（从哪个 checkpoint 续） ==="
tail -3 "$J/dl/train-dflash2.log" | cut -c1-140
ls -t "$J/data/dflash2_ckpts/"*.pt 2>/dev/null | head -2 | sed 's/^/  ckpt: /'
echo
echo "=== 2) 停训练（PowerShell .Kill()，历史上 taskkill 对卡在 CUDA 的进程无效） ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' } | ForEach-Object { '  killing pid ' + \$_.ProcessId; \$_.Kill() }" 2>/dev/null | tr -d '\r'
sleep 8
echo "  复查 python 进程："
powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train_dflash2*' }).Count" 2>/dev/null | tr -d '\r' | sed 's/^/  train_dflash2 剩余: /'
echo
echo "=== 3) 显存是否回落（必须回到 ~0 才能起引擎） ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
sleep 10
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  10 秒后: /'
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  app: /'
