#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== srv_cap.out 尾 15 ==="
tail -15 $J/dl/srv_cap.out 2>/dev/null
echo
echo "=== trace 最后 3 行 ==="
tail -3 $J/dl/mem_trace.log 2>/dev/null
echo
echo "=== scope 是否还在 / 峰值 ==="
M=/sys/fs/cgroup/user.slice/user-1001.slice/user@1001.service/app.slice/vllmcap.scope
if [ -d "$M" ]; then
  awk '{printf "memory.peak = %.2f GB\n", $1/1073741824}' "$M/memory.peak" 2>/dev/null
  cat "$M/memory.events" 2>/dev/null
else
  echo "  scope 已消失（unit 结束，systemd 已回收）"
fi
echo
echo "=== dmesg 里的 OOM 记录（最近） ==="
dmesg 2>/dev/null | tail -30 | grep -iE 'oom|killed process|memory cgroup out of memory' | head -6 || echo "  (dmesg 无权限或无 OOM 行)"
echo
echo "=== 宿主看门狗尾 3 ==="
tail -3 $J/dl/watchdog2.log 2>/dev/null
echo "=== 进程 ==="
pgrep -f 'inner_cap|vllm.entrypoints' >/dev/null && echo "仍在跑" || echo "已结束"
