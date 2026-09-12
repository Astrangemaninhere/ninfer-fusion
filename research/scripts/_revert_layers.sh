#!/bin/bash
# 回退层号实验（第二原则：可干净回退）+ 重编
set -u
F=/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h
export PATH="/home/user/.local/bin:$PATH"
cp -f /home/user/config_h.bak_before_feature_layers "$F" && echo '已回退层号实验'
grep -n 'target_feature_layers' "$F" | cut -c1-130
cd /home/user/ninfer-fusion || exit 1
T=$(grep -rl --include='*.cpp' --include='*.cu' -e 'impl/package.h' -e 'variant.h' src/targets/ 2>/dev/null | sort -u)
echo "touch $(echo "$T" | wc -l) 个包含者"
[ -n "$T" ] && touch $T
cd /home/user/ninfer-fusion/build || exit 2
MIN_FREE_GB=0 PER_JOB_GB=3 RESERVE_GB=1 MAX_JOBS=8 bash /mnt/c/Users/User/Documents/ziqinzhang/_par_build.sh ninfer ninfer-serve > /tmp/rev_build.log 2>&1
echo "build rc=$?"
tail -4 /tmp/rev_build.log | cut -c1-130
ls -l --time-style=+%H:%M /home/user/ninfer-fusion/build/apps/ninfer | awk '{print "  ninfer:", $6, $5}'
