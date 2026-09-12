#!/bin/bash
echo "=== 强杀僵死训练进程 pid 16344（.Kill() 已证明无效） ==="
powershell.exe -NoProfile -Command "Stop-Process -Id 16344 -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; if (Get-Process -Id 16344 -ErrorAction SilentlyContinue) { 'still alive after Stop-Process' } else { 'gone after Stop-Process' }" 2>/dev/null | tr -d '\r'
sleep 3
powershell.exe -NoProfile -Command "taskkill /F /PID 16344 2>&1 | Out-String" 2>/dev/null | tr -d '\r' | head -3
sleep 5
echo
echo "=== 复查进程与显存 ==="
powershell.exe -NoProfile -Command "(Get-Process -Id 16344 -ErrorAction SilentlyContinue) -eq \$null" 2>/dev/null | tr -d '\r' | sed 's/^/  pid16344 已消失: /'
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  显存: /'
sleep 20
nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  显存+20s: /'
echo
echo "=== 所有 python 进程（确认没别的残留） ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { '  pid ' + \$_.ProcessId + ' ' + \$_.CommandLine.Substring(0,[Math]::Min(60,\$_.CommandLine.Length)) }" 2>/dev/null | tr -d '\r' | head -5
