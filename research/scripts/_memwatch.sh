#!/bin/bash
# Sampler: record the build's memory curve so we can prove whether
# --split-compile-extended + the raised ceiling actually keep the peak under it.
# Detached (setsid), writes one line per 20s, stops after 90 samples (~30 min).
OUT=/mnt/c/Users/User/Documents/ziqinzhang/dl/memwatch.log
echo "=== memwatch $(date +%H:%M:%S) ===" >> "$OUT"
for i in $(seq 1 90); do
  used=$(free -m | awk '/^Mem:/{print $3}')
  avail=$(free -m | awk '/^Mem:/{print $7}')
  swap=$(free -m | awk '/^Swap:/{print $3}')
  top=$(ps -eo rss=,comm= --sort=-rss | head -3 | awk '{printf "%s=%dMB ", $2, $1/1024}')
  printf '%s used=%dMB avail=%dMB swap=%dMB  %s\n' "$(date +%H:%M:%S)" "$used" "$avail" "$swap" "$top" >> "$OUT"
  sleep 20
done
echo "=== memwatch end $(date +%H:%M:%S) ===" >> "$OUT"
