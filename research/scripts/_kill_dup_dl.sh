#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 杀掉修复前那个（16:42 启动，pid 41056，先核 cmdline） ==="
powershell.exe -NoProfile -Command "
\$p = Get-CimInstance Win32_Process -Filter \"ProcessId=41056\"
if (\$p -and \$p.CommandLine -like '*_hf_chunk.py*') { 'killing ' + \$p.ProcessId + ' start ' + \$p.CreationDate; \$p.Kill() } else { 'pid 41056 gone or not ours' }
" 2>/dev/null | tr -d '\r'
sleep 4
echo "=== 剩余下载器 ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*_hf_chunk.py*' } | ForEach-Object { '  pid ' + \$_.ProcessId + ' start ' + \$_.CreationDate.ToString('HH:mm:ss') }" 2>/dev/null | tr -d '\r'
echo
echo "=== 观察 60 秒：文件是否在长 ==="
for i in 1 2 3; do
  ls -l --time-style=+%H:%M:%S "$J/models/Spark-X2.5-4B/"model-0000*.safetensors 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
  sleep 20
done
echo "=== 日志尾 ==="
tail -3 "$J/dl/spark_weights.log" | tr -d '\r' | cut -c1-140
