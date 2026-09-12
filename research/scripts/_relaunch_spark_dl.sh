#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$J" || exit 3
echo "=== fix the fetcher ==="
python3 _fix_hf_chunk.py || exit 1
python3 -c "import py_compile; py_compile.compile('_hf_chunk.py', doraise=True); print('py_compile OK')"
echo
echo "=== stop the stalled single-GET run and clean its partial file ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*_hf_chunk.py*' } | ForEach-Object { 'killing pid ' + \$_.ProcessId; \$_.Kill() }" 2>/dev/null | tr -d '\r'
sleep 3
ls -l "$J/models/Spark-X2.5-4B/"*.safetensors 2>/dev/null | awk '{print "  partial:", $5, $NF}'
rm -f "$J/models/Spark-X2.5-4B/"model-*.safetensors
echo "  partial removed"
echo
echo "=== relaunch with sizes known ==="
powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Python312\python.exe' -ArgumentList '-u','C:\Users\User\Documents\ziqinzhang\_hf_chunk.py','XHToken/Spark-X2.5-4B','C:\Users\User\Documents\ziqinzhang\models\Spark-X2.5-4B','--include-suffix','.safetensors' -RedirectStandardOutput 'C:\Users\User\Documents\ziqinzhang\dl\spark_weights.log' -RedirectStandardError 'C:\Users\User\Documents\ziqinzhang\dl\spark_weights.err' -WindowStyle Hidden" 2>&1 | tr -d '\r'
sleep 35
tail -6 "$J/dl/spark_weights.log" 2>/dev/null | cut -c1-140
tail -3 "$J/dl/spark_weights.err" 2>/dev/null
