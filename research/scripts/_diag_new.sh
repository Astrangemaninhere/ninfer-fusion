#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== dl/ 里最新的 vref 相关文件 ==="
ls -lt --time-style=+%H:%M $J/dl/vref_* 2>/dev/null | head -10
echo
echo "=== 最新那份 err 的根因 ==="
NEW=$(ls -t $J/dl/vref_*.err 2>/dev/null | head -1)
echo "--- $NEW"
grep -nE 'RuntimeError|Error:|Traceback|OutOfMemory|out of memory|Killed|cuMemAlloc|FlashInfer' "$NEW" 2>/dev/null | tail -8
tail -6 "$NEW" 2>/dev/null
echo
echo "=== 孤儿 server 是谁（查 cmdline） ==="
for p in $(pgrep -f 'vllm.entrypoints' 2>/dev/null); do
  echo "pid=$p  $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"
done
echo
echo "=== dmesg 里是否有 OOM killer（最近） ==="
dmesg 2>/dev/null | tail -12 | grep -iE 'oom|killed|memory' | head -6 || echo "(dmesg 无权限或无记录)"
