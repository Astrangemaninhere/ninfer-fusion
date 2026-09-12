#!/bin/bash
# Empirical alignment run: does the target's own argmax agree with our draft?
# Prints (per round, per column) verify / draft / argmax, then the analyzer decides
# whether the acceptance kernel or the draft itself is at fault.
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/df2_align.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== df2 align run $(date '+%F %H:%M:%S') ==="
ls -d /home/user/models/*dflash2* 2>/dev/null | head -3
M=$(ls -d /home/user/models/*dflash2*.ninfer 2>/dev/null | head -1)
if [ -z "$M" ]; then echo "NO_DFLASH2_ARTIFACT"; exit 3; fi
echo "artifact: $M"
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd $R/build || exit 4
echo "--- probe run ---"
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 \
  --max-context 4096 --no-thinking --greedy --spec auto > $J/dl/df2_align_stdout.log 2> $J/dl/df2_probe_raw.log
echo "probe rc=$? lines=$(wc -l < $J/dl/df2_probe_raw.log)"
echo "--- plain greedy ground truth (no spec) ---"
timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 96 \
  --max-context 4096 --no-thinking --greedy --print-token-ids > $J/dl/df2_plain.log 2>&1
echo "plain rc=$?"
grep -E '^tokens[[:space:]]+generated ids' $J/dl/df2_plain.log | tail -1 | cut -c1-400
echo "--- spec summary from stdout ---"
grep -oE 'spec_accept_rate=[0-9.]+|spec_accepted=[0-9]+|spec_drafted=[0-9]+' $J/dl/df2_align_stdout.log | tail -6
grep -oE 'accepted by pos[[:space:]]+[0-9,]+' $J/dl/df2_align_stdout.log | tail -2
echo DF2_ALIGN_RUN_DONE
