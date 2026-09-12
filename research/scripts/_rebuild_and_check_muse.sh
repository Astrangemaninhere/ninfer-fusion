#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== Muse 验收脚本是否还在跑（决定能不能改它） ==="
pgrep -af '_muse_serve_accept.sh' | cut -c1-80 || echo "  已结束（可以修）"
echo
echo "=== Muse 验收脚本里试了哪些 KV 档 ==="
grep -nE 'for kv|kv in|nvfp4|bf16|i8|e8' "$J/_muse_serve_accept.sh" 2>/dev/null | head -14 | cut -c1-140
echo
echo "=== 起重建（layouts_impl.h 变更 → variant TU 重编 + 链接） ==="
setsid nohup bash "$J/_rebuild_after_s35.sh" > /dev/null 2>&1 < /dev/null &
sleep 20
tail -4 "$J/dl/rebuild_after_s35.log" | cut -c1-140
