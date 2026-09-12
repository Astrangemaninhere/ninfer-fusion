#!/bin/bash
G=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/gui
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== model_import.py 是不是自带页面的服务端？（决定它该不该有语言开关） ==="
grep -nE 'BaseHTTPRequestHandler|ThreadingHTTPServer|def do_GET|def do_POST|PAGE =|render_page|__main__' "$G/model_import.py" | head -12 | cut -c1-120
echo
echo "=== gui_tips.py 是否只是字典模块 ==="
grep -nE 'def |PAGE|HTTP' "$G/gui_tips.py" | head -6 | cut -c1-120
echo
echo "=== Spark 下载还活着吗 ==="
tail -3 "$J/dl/spark_weights.log" | tr -d '\r' | cut -c1-140
ls -l --time-style=+%H:%M "$J/models/Spark-X2.5-4B/"model-0000*.safetensors 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
date '+  现在 %H:%M:%S'
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*_hf_chunk.py*' } | ForEach-Object { '  下载器存活 pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r'
tail -2 "$J/dl/spark_weights.err" 2>/dev/null | cut -c1-140
