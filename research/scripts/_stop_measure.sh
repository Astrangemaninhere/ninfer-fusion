#!/bin/bash
# 停掉正在跑的测量（按 PID，先核 cmdline），准备修脚本
J=/mnt/c/Users/User/Documents/ziqinzhang
for p in $(pgrep -f '_post_build_measure.sh'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline)
  case "$cmd" in *_post_build_measure.sh*) echo "  TERM watcher $p"; kill -TERM "$p" ;; esac
done
for p in $(pgrep -f 'ninfer-serve'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline)
  case "$cmd" in *ninfer-serve*) echo "  TERM serve $p"; kill -TERM "$p" ;; esac
done
sleep 4
pgrep -af 'ninfer-serve|_post_build_measure' | cut -c1-80 || echo "  都停了"
echo
echo "=== 现场证据：plain 那次请求的真实返回（前 400 字符） ==="
head -c 400 /tmp/pb_resp_plain_zh.json 2>/dev/null; echo
echo "=== 引擎行 ==="
grep -E 'done .*gen=' /home/user/pb_plain.log 2>/dev/null | tail -1 | cut -c1-220
echo
echo "=== 结论：content 为空 = thinking 模式吃掉了输出（脚本没传 --no-thinking） ==="
grep -o '"reasoning_content"[^,]*' /tmp/pb_resp_plain_zh.json 2>/dev/null | head -c 200 || echo "  （响应里没有 reasoning_content 字段）"
