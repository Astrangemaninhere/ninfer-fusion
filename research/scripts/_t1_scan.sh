#!/bin/bash
# scan /home/user/*.log (+ .out) for accepted-by-pos profiles with timestamps
cd /home/user || exit 1
for f in *.log *.out; do
  [ -f "$f" ] || continue
  prof=$(grep -a -m1 'accepted by pos' "$f" 2>/dev/null | sed 's/.*accepted by pos *//')
  [ -n "$prof" ] || continue
  rounds=$(grep -a -m1 'dflash rounds' "$f" 2>/dev/null | sed 's/.*rounds *//')
  drafted=$(grep -a -m1 'dflash drafted tokens' "$f" 2>/dev/null | sed 's/.*tokens *//')
  acc=$(grep -a -m1 'dflash accepted tokens' "$f" 2>/dev/null | sed 's/.*tokens *//')
  ts=$(date -r "$f" +%m-%d_%H:%M)
  printf '%s|%-32s|r=%-5s|d=%-6s|a=%-5s|%s\n' "$ts" "$f" "$rounds" "$drafted" "$acc" "$prof"
done | sort
