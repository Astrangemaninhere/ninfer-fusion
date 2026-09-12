#!/bin/bash
echo "=== 两个下载器进程的细节 ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*_hf_chunk.py*' } | ForEach-Object { '{0}  start={1}  ppid={2}  {3}' -f \$_.ProcessId, \$_.CreationDate.ToString('HH:mm:ss'), \$_.ParentProcessId, \$_.CommandLine.Substring(0,[Math]::Min(130,\$_.CommandLine.Length)) }" 2>/dev/null | tr -d '\r'
echo
echo "=== 谁在写文件的句柄（按最近修改时间判断） ==="
ls -l --time-style=+%H:%M:%S /mnt/c/Users/User/Documents/ziqinzhang/models/Spark-X2.5-4B/*.safetensors 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
sleep 12
echo "  12 秒后再看："
ls -l --time-style=+%H:%M:%S /mnt/c/Users/User/Documents/ziqinzhang/models/Spark-X2.5-4B/*.safetensors 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
