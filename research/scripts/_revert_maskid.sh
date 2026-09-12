#!/bin/bash
# 回退 mask-id 实验 + 重编（保持树干净），并打印当前 dspark/dflash2 基线
set -u
R=/home/user/ninfer-fusion
F=$R/src/targets/qwen3_6_27b/impl/config.h
export PATH="/home/user/.local/bin:$PATH"
cp -f /home/user/config_h.bak_maskid "$F" && echo '已回退 mask-id 实验'
grep -n 'mask_token' "$F" | cut -c1-120
cd "$R" || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
[ -n "$T" ] && touch $T
cd "$R/build" || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve > /tmp/rev_maskid.log 2>&1
echo "  build rc=$?"
tail -2 /tmp/rev_maskid.log | cut -c1-110
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" | awk '{print "  ninfer:", $6, $5}'
echo
echo '=== 实验结论汇总（dspark K=7）==='
echo '  mask=248077（原）: 7.38%  pos=[18,0,0,0,0,0,0]'
echo '  mask=190221（实验）: 6.83%  pos=[16,1,0,0,0,0,0]  => 无改善'
