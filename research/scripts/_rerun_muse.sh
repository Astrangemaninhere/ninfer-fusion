#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 停 K4（避免它在我跑 Muse 验收时启动训练抢显存） ==="
for p in $(pgrep -f '_window_k4.sh'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline)
  case "$cmd" in *_window_k4.sh*) echo "  TERM K4 $p"; kill -TERM "$p" ;; esac
done
sleep 2
pgrep -af '_window_k4' | cut -c1-60 || echo "  K4 已停"
echo
echo "=== 2) 确认 GPU 空闲 ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pgrep -af 'ninfer-serve' | cut -c1-80 || echo "  无 serve"
echo
echo "=== 3) 用新二进制跑 Muse 验收（detached） ==="
: > "$J/dl/muse_accept_v2.log"
setsid nohup bash -c "bash '$J/_muse_serve_accept.sh' >> '$J/dl/muse_accept_v2.log' 2>&1; echo MUSE_ACCEPT_V2_DONE >> '$J/dl/muse_accept_v2.log'" > /dev/null 2>&1 < /dev/null &
sleep 25
tail -8 "$J/dl/muse_accept_v2.log" 2>/dev/null | cut -c1-150
