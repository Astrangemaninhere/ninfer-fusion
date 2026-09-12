#!/bin/bash
# 跑带深度探针的 dump（logits / proposal_hidden / projected）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
cd /home/user/ninfer-fusion/build || exit 3
pkill -9 -x ninfer 2>/dev/null; sleep 2
rm -f "$J"/dl/df2scores_*.bin
NINFER_DF2SCORES=1 NINFER_DF2DBG=1 timeout 900 ./apps/ninfer \
  /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt '请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。' \
  --max-new 96 --max-context 4096 --no-thinking --greedy --spec dflash2 --no-cuda-graph \
  > "$J/dl/s4_scores.log" 2>&1
echo "rc=$?"
for t in scores cand unary front anch logits ph proj; do
  printf "  %-8s %s 个\n" "$t" "$(ls "$J"/dl/df2scores_${t}_*.bin 2>/dev/null | wc -l)"
done
ls -l "$J"/dl/df2scores_logits_1.bin "$J"/dl/df2scores_ph_1.bin "$J"/dl/df2scores_proj_1.bin 2>/dev/null | awk '{print "  " $5, $9}'
grep -m1 'accepted by pos' "$J/dl/s4_scores.log" || true
