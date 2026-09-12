#!/bin/bash
# 用 Windows python 跑 ids16 口径探针（Windows 侧 python 认得 C:\ 路径）
powershell.exe -NoProfile -Command "Start-Process -FilePath 'C:\Program Files\Python312\python.exe' -ArgumentList '-u','C:\Users\User\Documents\ziqinzhang\_probe_ids16_semantics.py' -RedirectStandardOutput 'C:\Users\User\Documents\ziqinzhang\dl\ids16_probe.log' -RedirectStandardError 'C:\Users\User\Documents\ziqinzhang\dl\ids16_probe.err' -WindowStyle Hidden" 2>&1 | tr -d '\r'
sleep 35
echo "=== 结果 ==="
cat /mnt/c/Users/User/Documents/ziqinzhang/dl/ids16_probe.log 2>/dev/null | head -20
echo "=== err ==="
tail -4 /mnt/c/Users/User/Documents/ziqinzhang/dl/ids16_probe.err 2>/dev/null | tr -d '\r'
