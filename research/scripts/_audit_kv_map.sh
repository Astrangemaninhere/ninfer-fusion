#!/bin/bash
# ensure_sequence_kv_mapped 的签名与全部调用点：第三参在 dflash 是 frontier、dflash2 是 0，
# 这处兄弟不一致是有意（验证需要全前缀）还是遗留？
R=/home/user/ninfer-fusion
echo '=== 声明 ==='
grep -rn 'ensure_sequence_kv_mapped' "$R/src" | grep -vE '^\s*$' | cut -c1-150
echo
echo '=== 定义体（含参数名） ==='
for f in $(grep -rl 'void ensure_sequence_kv_mapped\|ensure_sequence_kv_mapped(' "$R/src" --include=*.h --include=*.cpp | sort -u); do
  echo "--- $f ---"
  awk '/ensure_sequence_kv_mapped\(/{c=NR} c && NR>=c && NR<c+22' "$f" | head -26 | cut -c1-135
done
