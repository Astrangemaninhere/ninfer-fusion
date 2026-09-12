#!/bin/bash
echo "=== E3 scratch ==="
ls -lR /tmp/e3s45b 2>/dev/null | head -30
echo "=== E3 candidate vs mirror (prefill) ==="
if [ -f /tmp/e3s45b/out/gqa_attention_prefill.cu ]; then
  diff -u /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/src/ops/launcher/gqa_attention_prefill.cu \
          /tmp/e3s45b/out/gqa_attention_prefill.cu | head -80
else
  echo "(no candidate yet)"
fi
echo "=== memory detail ==="
free -m | head -3
echo "=== nvcc RSS ==="
for p in $(pgrep -f bin/nvcc); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline | grep -oE '[^ ]+\.cu' | tail -1)
  rss=$(awk '/VmRSS/{print $2/1024" MB"}' /proc/$p/status 2>/dev/null)
  echo "  pid=$p rss=$rss tu=$cmd"
done
echo "=== v2 progress ==="
tail -4 /mnt/c/Users/User/Documents/ziqinzhang/dl/df2_w9_v2.log
