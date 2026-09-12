#!/bin/bash
# 清零提案头后跑一次，与 refhead（4.81%）对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
exec > >(tee -a "$J/dl/arm_zerodrafthead.log") 2>&1
echo "=== 清零提案头跑 $(date '+%H:%M:%S') ==="
cd "$R/build" || exit 3
timeout 1200 ./apps/ninfer "$ART" --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --spec dflash2 --print-token-ids > $J/dl/fx_zerodrafthead.log 2>&1
echo "  rc=$?"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|dflash2 drafted tokens|dflash2 rounds|decode speed|fallback' $J/dl/fx_zerodrafthead.log | sed 's/^/  /'
echo
echo "=== 对照 ==="
printf "  %-16s %-10s %s\n" 臂 接受率 位置剖面
for t in refhead zerodrafthead; do
  f=$J/dl/fx_$t.log; [ "$t" = refhead ] && f=$J/dl/arm_refhead.log
  a=$(grep -m1 'dflash2 acceptance rate' $f 2>/dev/null | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $f 2>/dev/null | sed 's/.*pos *//')
  printf "  %-16s %-10s %s\n" "$t" "${a:-?}" "${p:-?}"
done
echo ZERO_DRAFTHEAD_RUN_DONE
