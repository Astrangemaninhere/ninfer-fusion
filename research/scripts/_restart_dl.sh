#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== restart the chunked FlashNext/MiniCPM download (Windows python, detached) ==="
powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Python312\python.exe' -ArgumentList '-u','C:\Users\User\Documents\ziqinzhang\_hf_chunked.py' -RedirectStandardOutput 'C:\Users\User\Documents\ziqinzhang\dl\hf-chunk.out' -RedirectStandardError 'C:\Users\User\Documents\ziqinzhang\dl\hf-chunk.err' -WindowStyle Hidden" 2>&1 | tr -d '\r'
sleep 25
echo "--- hf-chunk.log tail ---"
tail -5 "$J/dl/hf-chunk.log" 2>/dev/null | cut -c1-140
echo "--- process check ---"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*chunked*' } | ForEach-Object { 'running pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r'
echo "--- current FlashNext size ---"
du -sh "$J/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4" 2>/dev/null
