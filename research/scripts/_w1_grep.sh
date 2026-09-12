#!/bin/bash
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
echo "=== grep 非因果/non-causal/causal across _collab (md only, with file:line):"
grep -rn --include=*.md -e '非因果' -e 'non-causal' -e 'noncausal' -e '因果' "$C" 2>/dev/null | grep -viE 'causal_gqa_row|noncausal_gqa' | head -60
echo
echo "=== grep 'W1' task definition in board.md:"
grep -n 'W1\b' "$C/board.md" | head -20
echo
echo "=== list of _collab/*.md newest 30:"
ls -t "$C"/*.md 2>/dev/null | head -30
