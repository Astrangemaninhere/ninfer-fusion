#!/bin/bash
# E2: maximally copyable context. The target will continue the digit pattern verbatim,
# so a healthy drafter must reproduce it and acceptance must be near-maximal. If the
# drafts degenerate instead, the draft block's context is not reaching it.
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/df2_copy.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== df2 copy-context run $(date '+%F %H:%M:%S') ==="
M=$(ls -d /home/user/models/*dflash2*.ninfer 2>/dev/null | head -1)
[ -z "$M" ] && { echo NO_ARTIFACT; exit 3; }
P='0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2'
cd $R/build || exit 4
echo "--- probe run (digit pattern) ---"
NINFER_DF2DBG=1 timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 64 \
  --max-context 4096 --no-thinking --greedy --spec auto \
  > $J/dl/df2_copy_stdout.log 2> $J/dl/df2_copy_probe.log
echo "probe rc=$? probe_lines=$(wc -l < $J/dl/df2_copy_probe.log)"
echo "--- plain greedy (no spec) ---"
timeout 900 ./apps/ninfer "$M" --prompt "$P" --max-new 64 \
  --max-context 4096 --no-thinking --greedy --print-token-ids > $J/dl/df2_copy_plain.log 2>&1
echo "plain rc=$?"
grep -E '^tokens[[:space:]]+generated ids' $J/dl/df2_copy_plain.log | tail -1 | cut -c1-300
echo DF2_COPY_DONE
