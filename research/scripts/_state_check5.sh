#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== disk ==="
df -h /mnt/c | tail -1
echo "=== FlashNext download progress ==="
tail -3 "$J/dl/hf-chunk.log" | cut -c1-130
du -sh "$J/models/Qwen3.8-Flash-Next-ABLITERATED-NVFP4" 2>/dev/null
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*chunked*' } | ForEach-Object { 'downloader pid ' + \$_.ProcessId }" 2>/dev/null | tr -d '\r'
echo
echo "=== build / measurement ==="
grep -oE '^\[[ 0-9]+%\] (Building|Linking)[^"]*' /tmp/pa_make_1.log | tail -2
for p in $(pgrep -f bin/nvcc); do echo "  nvcc $(ps -o etime= -p $p | tr -d ' ')"; break; done
grep -c 'patchA build done' "$J/dl/patchA_build.log" 2>/dev/null | sed 's/^/  patchA build done markers: /'
tail -1 "$J/dl/postbuild_measure.log" 2>/dev/null | cut -c1-110
echo
echo "=== Spark metadata on disk ==="
du -sh "$J/models/Spark-X2.5-4B" 2>/dev/null
ls "$J/models/Spark-X2.5-4B" | head -6 | tr '\n' ' '
echo
echo "=== agents (GUI) ==="
ls -lt /mnt/c/Users/User/.zcode/cli/agents/sess_2cd5cec3-4f5e-4c0e-a33a-a68c2ac54c5f/ 2>/dev/null | head -4 | cut -c1-60
