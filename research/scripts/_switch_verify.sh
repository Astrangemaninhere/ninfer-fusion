#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 停掉旧的验证链（会与新脚本抢 GPU） ==="
for p in $(pgrep -f '_offset_family_verify'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cmd" in *_offset_family_verify*) echo "  TERM $p"; kill -TERM "$p" 2>/dev/null ;; esac
done
sleep 2
pgrep -af '_offset_family_verify' >/dev/null && echo "  仍有残留" || echo "  已停"
echo
echo "=== 起新的对照链（等编译 idle，然后跑三组前/后对照） ==="
setsid nohup bash "$J/_post_fix_verify.sh" > /dev/null 2>&1 < /dev/null &
sleep 5
pgrep -af '_post_fix_verify' | head -1 | cut -c1-60
echo
echo "=== 当前编译状态 ==="
tail -3 "$J/dl/rebuild_after_s35.log" | cut -c1-135
for p in $(pgrep -f 'bin/nvcc|cc1plus'); do echo "  在编 $(ps -o etime= -p $p | tr -d ' ')"; break; done
