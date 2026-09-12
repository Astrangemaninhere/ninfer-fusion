#!/bin/bash
# 重新生成 8 类 dump 并复核"深度签名"（top-1 游程 / 相邻深度 Jaccard），与参考对比
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang/dl
export PATH=/home/user/.local/bin:$PATH
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
rm -f "$J"/df2scores_*.bin
NINFER_DF2SCORES=1 NINFER_DF2DBG=1 timeout 900 ./apps/ninfer \
  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph \
  > "$J/regen_scores.log" 2>&1
echo "rc=$?"
for t in scores cand unary front anch logits ph proj; do
  printf "  %-8s %s 组\n" "$t" "$(ls "$J"/df2scores_${t}_*.bin 2>/dev/null | wc -l)"
done
grep -m1 'accepted by pos' "$J/regen_scores.log" || true
grep -c 'df2dbg] row=' "$J/regen_scores.log" || true
