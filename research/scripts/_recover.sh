#!/bin/bash
# 恢复后：清掉可能残留的 vllm 进程/抢占端口，确认机器健康
echo "=== WSL 状态 ==="
uptime
free -g | head -2
df -h / | tail -1
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -2
echo
echo "=== 清理残留 vllm server（先看 cmdline 再杀，铁律②） ==="
n=0
for p in $(pgrep -f 'vllm.entrypoints' 2>/dev/null); do
  cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cl" in *vllm.entrypoints*) echo "  kill $p"; kill "$p" 2>/dev/null; n=$((n+1));; esac
done
echo "  清掉 $n 个"
echo
echo "=== 端口 8127-8133 是否还被占 ==="
for port in 8127 8129 8130 8131 8132 8133; do
  if curl -s -m 2 -o /dev/null http://127.0.0.1:$port/v1/models 2>/dev/null; then echo "  $port 还在应答"; fi
done
echo "(无输出=端口都干净)"
echo
echo "=== 我留在 WSL 的东西（供你决定是否删） ==="
du -sh /home/user/vllm029 2>/dev/null
du -sh /home/user/models/draft_dflash2_ref 2>/dev/null
du -sh /home/user/models 2>/dev/null
