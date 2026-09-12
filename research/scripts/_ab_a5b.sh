#!/bin/bash
# A5b 的干净 A/B：保存当前(已回退)二进制 -> 重新应用 A5b -> 重编 -> 得到一对只差 A5b 的二进制
set -u
R=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
export PATH="/home/user/.local/bin:$PATH"

echo '=== 1) 保存 A5b-已回退 的二进制 ==='
cp -f "$R/build/apps/ninfer" /home/user/ninfer_no_a5b
ls -l --time-style=+%H:%M /home/user/ninfer_no_a5b | awk '{print "  ninfer_no_a5b:", $6, $5}'
echo "  md5: $(md5sum /home/user/ninfer_no_a5b | cut -c1-16)"

echo
echo '=== 2) 重新应用 A5b ==='
D=$C/A5b_attention_valid_width.diff
tr -d '\r' < "$D" > /tmp/a5b_fwd.diff
patch -p1 -d "$R" < /tmp/a5b_fwd.diff && echo "  已应用"
grep -n 'set_i32_scalar(attention_valid' "$R/src/targets/qwen3_6/impl/runtime/dflash_impl.h" | cut -c1-120

echo
echo '=== 3) touch + 重编（得到 A5b-已应用 的二进制）==='
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'dflash_impl.h' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
N=$(printf '%s\n' "$T" | grep -c . || true)
echo "  touch $N"
[ -n "$T" ] && touch $T
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve > /tmp/a5b_build.log 2>&1
echo "  build rc=$?"
tail -3 /tmp/a5b_build.log | cut -c1-120
cp -f "$R/build/apps/ninfer" /home/user/ninfer_with_a5b
ls -l --time-style=+%H:%M /home/user/ninfer_with_a5b | awk '{print "  ninfer_with_a5b:", $6, $5}'
echo "  md5: $(md5sum /home/user/ninfer_with_a5b | cut -c1-16)"
echo
echo '=== 4) 两个二进制是否真的不同 ==='
if cmp -s /home/user/ninfer_no_a5b /home/user/ninfer_with_a5b; then
  echo "  警告：两者相同 —— A5b 可能没改到代码或没重编"
else
  echo "  OK：两者不同，A/B 有效"
fi
