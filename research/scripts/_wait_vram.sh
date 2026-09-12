#!/bin/bash
for i in 1 2 3 4 5 6; do
  sleep 15
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ' ')
  echo "  [$((i*15))s] used=$used  apps=[$apps]"
  case "$used" in
    *" MiB"*) v=${used%% *};; *) v=99999;;
  esac
  [ "$v" -lt 2000 ] && { echo "  显存已回落"; break; }
done
echo
echo "=== 若有残留 python，再杀一次 ==="
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { '  pid ' + \$_.ProcessId + ' ' + \$_.CommandLine.Substring(0,[Math]::Min(70,\$_.CommandLine.Length)) }" 2>/dev/null | tr -d '\r' | head -6
