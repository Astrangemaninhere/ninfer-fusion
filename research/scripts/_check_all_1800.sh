#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== Spark 权重下载是否还在推进 ==="
tail -2 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-120
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*_hf_chunk.py*' } | ForEach-Object { '  downloader alive pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r' || echo "  （下载器不在跑）"
echo
echo "=== 编译进度 ==="
tail -3 /tmp/reb_ninfer_1.log 2>/dev/null | cut -c1-140
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
echo
echo "=== 训练 ==="
tail -2 "$J/dl/train-dflash2.log" 2>/dev/null | cut -c1-140
ls -t "$J/data/dflash2_ckpts/"*.pt 2>/dev/null | head -2 | sed 's/^/  ckpt: /'
