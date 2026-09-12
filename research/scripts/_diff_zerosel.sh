#!/bin/bash
# 差分测试：清零选择器码本后跑一次，与 refhead 对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
S=$J/data/dflash2_zeroselector.ninfer
D=/home/user/models/qwen3_8_27b_nvfp4_dflash2_zeroselector.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LOG=$J/dl/diff_zeroselector.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 差分测试（清零码本） $(date '+%H:%M:%S') ==="

echo "--- 拷贝到原生盘 ---"
if [ ! -f "$D" ] || [ "$(stat -c %s "$D" 2>/dev/null)" != "$(stat -c %s "$S")" ]; then
  time cp -f "$S" "$D"
fi
echo "  src=$(stat -c %s "$S")  dst=$(stat -c %s "$D")"

echo "--- 跑 ---"
cd "$R/build" || exit 3
NINFER_DF2DBG=1 timeout 1200 ./apps/ninfer "$D" \
   --prompt "$P" --max-new 96 --max-context 4096 \
   --no-thinking --greedy --spec dflash2 --print-token-ids \
   > $J/dl/arm_zeroselector.log 2>&1
echo "  rc=$? lines=$(grep -c df2dbg $J/dl/arm_zeroselector.log)"
grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|dflash2 drafted tokens|dflash2 rounds|decode speed|kv cache dtype' $J/dl/arm_zeroselector.log | sed 's/^/  /'

echo
echo "=== 三臂对比 ==="
printf "  %-14s %-10s %-22s %s\n" 臂 接受率 位置剖面 长度
for tag in control refhead zeroselector; do
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/arm_$tag.log 2>/dev/null | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/arm_$tag.log 2>/dev/null | sed 's/.*pos *//')
  l=$(grep -m1 'dflash2 acceptance length' $J/dl/arm_$tag.log 2>/dev/null | grep -oE '[0-9.]+ tok/round')
  printf "  %-14s %-10s %-22s %s\n" "$tag" "${a:-?}" "${p:-?}" "${l:-?}"
done
echo DIFF_ZEROSEL_DONE
