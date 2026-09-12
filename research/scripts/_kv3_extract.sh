#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "arm | 模型档 | 接受率 | 位置剖面 | 解码 | 轮数 | 起草/接受"
for f in $J/dl/kv3_*.log; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .log); b=${b#kv3_}
  kind=$(grep -m 1 -E 'mtp acceptance rate|dflash2 acceptance rate|dflash acceptance rate' "$f" | grep -oE '(mtp|dflash2|dflash) acceptance rate' | awk '{print $1}')
  a=$(grep -m 1 -E 'mtp acceptance rate|dflash2 acceptance rate|dflash acceptance rate' "$f" | grep -oE '[0-9.]+%')
  p=$(grep -m 1 -E 'mtp accepted by pos|dflash2 accepted by pos|dflash accepted by pos' "$f" | sed 's/.*pos *//')
  sp=$(grep -m 1 'decode speed' "$f" | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m 1 -E 'mtp rounds|dflash2 rounds|dflash rounds' "$f" | grep -oE '[0-9]+')
  dr=$(grep -m 1 -E 'mtp drafted tokens|dflash2 drafted tokens|dflash drafted tokens' "$f" | grep -oE '[0-9]+')
  ac=$(grep -m 1 -E 'mtp accepted tokens|dflash2 accepted tokens|dflash accepted tokens' "$f" | grep -oE '[0-9]+')
  printf "%-16s %-9s %-8s %-22s %-12s %-5s %s/%s\n" "$b" "${kind:-?}" "${a:-?}" "${p:-?}" "${sp:-?}" "${rd:-?}" "${dr:-?}" "${ac:-?}"
done
echo
echo "=== 若某臂报错，打印其首行 ==="
for f in $J/dl/kv3_*.log; do
  [ -f "$f" ] || continue
  if ! grep -q 'acceptance rate' "$f"; then echo "--- $(basename $f) ---"; head -3 "$f"; fi
done
