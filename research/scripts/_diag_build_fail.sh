#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
Q=$R/_orig_quarantine
echo "=== [1] 停掉 K3（它在等一个不会来的标记，之后会用旧二进制跑判定） ==="
for p in $(pgrep -f '_window_k3.sh'); do
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline)
  case "$cmd" in *_window_k3.sh*) echo "  TERM $p"; kill -TERM "$p";; esac
done
sleep 2
pgrep -af '_window_k3' || echo "  K3 已停"
echo
echo "=== [2] 测量守护：它在等 ninfer-serve 变新（也会一直等）——先留着，稍后重跑 ==="
tail -1 "$J/dl/postbuild_measure.log" | cut -c1-100
echo
echo "=== [3] 编译错误现场 ==="
sed -n '1720,1732p' "$R/src/runtime/engine/engine_core.h" | cat -n | sed 's/^/  1720+/'
echo "  --- 1800-1806 ---"
sed -n '1798,1806p' "$R/src/runtime/engine/engine_core.h" | cat -n | sed 's/^/  1798+/'
echo "  --- resource_manager.h:233-240 ---"
sed -n '233,240p' "$R/src/runtime/engine/resource_manager.h" | cat -n | sed 's/^/  233+/'
echo
echo "=== [4] 这三处是不是补丁改过的？（隔离区里有 engine_core.h.orig） ==="
ls -l "$Q/src/runtime/engine/" 2>/dev/null | awk '{print "  ", $5, $NF}'
if [ -f "$Q/src/runtime/engine/engine_core.h.orig" ]; then
  echo "  --- diff .orig vs live（只看前 60 行差异） ---"
  diff -u "$Q/src/runtime/engine/engine_core.h.orig" "$R/src/runtime/engine/engine_core.h" | head -60 | cut -c1-140
fi
echo
echo "=== [5] 有没有别的 .orig 提示还有哪些 src 被改过 ==="
find "$Q" -name '*.orig' -printf '  %p\n' 2>/dev/null | sed "s#$Q/##"
