#!/bin/bash
# dflash2: full head vs draft head, with an explicit --spec (auto is not honoured
# together with --lm-head-draft / --draft-tokens).
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/lmhead_ab2.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== dflash2 explicit-spec head A/B $(date '+%F %H:%M:%S') ==="
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd $R/build || exit 4
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer

one() {
  local tag=$1; shift
  local log=/home/user/lmh2_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  echo "  [$tag] rc=$? n=$(grep -cE '^tokens' "$log")"
  grep -E 'proposal_head|lm_head' "$log" | head -2
  grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1
  grep -oE 'spec_accept_rate=[0-9.]+|spec_accepted=[0-9]+|spec_drafted=[0-9]+' "$log" | tail -3
  tail -3 "$log" | grep -iE 'error|throw|invalid|abort|what' || true
}
echo "--- dflash2 --spec dflash2 (full head) ---"
one df2_full  --spec dflash2
echo "--- dflash2 --spec dflash2 --lm-head-draft ---"
one df2_lmhd  --spec dflash2 --lm-head-draft
echo LMHEAD_AB2_DONE
