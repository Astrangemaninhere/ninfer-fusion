#!/bin/bash
# 决定性 A/B：现役 artifact（混血草稿头）vs 新 artifact（incoai 自洽草稿头）
# 同协议：greedy / no-thinking / bf16 KV / 同 prompt / max-new 96，并带逐列探针
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
OUT=$J/dl/ab_refhead.log
CTRL=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
TREAT=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
cd "$R/build" || exit 3
exec > >(tee -a "$OUT") 2>&1
echo "=== 草稿头 A/B $(date '+%F %H:%M:%S') ==="
for f in "$CTRL" "$TREAT"; do
  if [ -f "$f" ]; then echo "  OK   $f ($(stat -c %s "$f") B)"; else echo "  MISS $f"; exit 2; fi
done

run_arm(){  # $1=tag  $2=artifact
  local tag=$1 art=$2
  echo
  echo "########## [$tag] $(basename $art)"
  NINFER_DF2DBG=1 timeout 1200 ./apps/ninfer "$art" \
     --prompt "$P" --max-new 96 --max-context 4096 \
     --no-thinking --greedy --spec dflash2 --print-token-ids \
     > $J/dl/arm_$tag.log 2>&1
  echo "  rc=$?  lines=$(grep -c df2dbg $J/dl/arm_$tag.log)"
  grep -E 'dflash2 acceptance rate|dflash2 accepted by pos|dflash2 acceptance length|dflash2 drafted tokens|dflash2 rounds|dflash2 fallback|decode speed|kv cache dtype' $J/dl/arm_$tag.log | sed 's/^/  /'
}

run_arm control "$CTRL"
run_arm refhead "$TREAT"

echo
echo "=== 汇总 ==="
for tag in control refhead; do
  a=$(grep -m1 'dflash2 acceptance rate' $J/dl/arm_$tag.log | grep -oE '[0-9.]+%')
  p=$(grep -m1 'dflash2 accepted by pos' $J/dl/arm_$tag.log | sed 's/.*pos *//')
  l=$(grep -m1 'dflash2 acceptance length' $J/dl/arm_$tag.log | grep -oE '[0-9.]+ tok/round')
  s=$(grep -m1 'decode speed' $J/dl/arm_$tag.log | grep -oE '[0-9.]+ tok/s')
  echo "  $tag : 接受率 ${a:-?}  位置剖面 ${p:-?}  接受长度 ${l:-?}  解码 ${s}"
done
echo AB_REFHEAD_DONE
