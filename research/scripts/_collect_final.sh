#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 四路对比结果（K4 第 4 步） ==="
grep -E 'SERVE_FAILED|^\| |up|TEXT|decode=' "$J/dl/window_k4.log" 2>/dev/null | head -20 | cut -c1-150
echo
echo "=== 那档 plain_mtp 为何失败 ==="
for f in /home/user/s4w_plain_mtp.log; do
  echo "--- $f ---"; tail -6 "$f" 2>/dev/null | cut -c1-170
done
echo
echo "=== 训练状态 ==="
tail -4 "$J/dl/train-dflash2.log" 2>/dev/null | cut -c1-150
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"ProcessId=16344\" | ForEach-Object { '  training alive since ' + \$_.CreationDate.ToString('HH:mm:ss') }" 2>/dev/null | tr -d '\r'
ls -t "$J/data/dflash2_ckpts/"*.pt 2>/dev/null | head -3 | sed 's/^/  ckpt: /'
echo
echo "=== Spark 权重 ==="
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
tail -2 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-120
