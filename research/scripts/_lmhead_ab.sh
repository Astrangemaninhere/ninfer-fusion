#!/bin/bash
# Decisive A/B: does the dedicated draft head (--lm-head-draft) restore acceptance?
#   dspark  : our path HONOURS proposal_head -> expect a jump if the head is the cause
#   dflash2 : our path IGNORES proposal_head  -> expect no change (proves the gap)
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/lmhead_ab.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== lm-head-draft A/B $(date '+%F %H:%M:%S') ==="
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd $R/build || exit 4

one() {  # tag artifact extra...
  local tag=$1 art=$2; shift 2
  local log=/home/user/lmh_$tag.log
  timeout 900 ./apps/ninfer "/home/user/models/$art" --prompt "$P" --max-new 96 \
      --max-context 4096 --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  local rc=$?
  local acc; acc=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local ac;  ac=$(grep -oE 'spec_accepted=[0-9]+' "$log" | tail -1)
  local dr;  dr=$(grep -oE 'spec_drafted=[0-9]+' "$log" | tail -1)
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local ids; ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
                   sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' | wc -w)
  echo "  [$tag] rc=$rc n=$ids $ac $dr $acc pos=[$pos]"
  echo "$tag|$ac|$dr|$acc|$pos|$ids" >> /tmp/lmh_rows
}

: > /tmp/lmh_rows
echo "--- dspark K=7 : full head vs draft head ---"
one dspark_full  qwen3_8_27b_nvfp4_dspark.ninfer   --spec dflash --draft-tokens 7
one dspark_lmhd  qwen3_8_27b_nvfp4_dspark.ninfer   --spec dflash --draft-tokens 7 --lm-head-draft
echo "--- dflash2 K=7 : full head vs draft head (our path ignores the flag) ---"
one df2_full     qwen3_8_27b_nvfp4_dflash2.ninfer  --spec auto
one df2_lmhd     qwen3_8_27b_nvfp4_dflash2.ninfer  --spec auto --lm-head-draft
echo
echo "tag | accepted | drafted | rate | per-position | generated"
cat /tmp/lmh_rows
echo LMHEAD_AB_DONE
