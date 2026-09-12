#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang/dl
for f in clean_vllm.log abl_vs_clean.log clean_run.log; do
  p="$D/$f"
  [ -f "$p" ] || { echo "=== $f: MISSING ==="; continue; }
  echo "=== $f  ($(wc -l < "$p") lines) ==="
  grep -ain -m 12 'accept\|specdecode\|spec_decode\|draft\|method=' "$p" | head -12
  echo
done
