#!/bin/bash
# temp: W8 — rmsnorm weight-folding PPL comparison (baseline vs fold3 vs foldmlp).
# The TODO note applies: perplexity does not support --kv-dtype nvfp4, so defaults (bf16) are used.
set -u
BIN=/home/user/ninfer-fusion/build/apps/ninfer-perplexity
CORPUS=/home/user/perplexity-corpus-long.txt
OUT=/home/user/w8_ppl
mkdir -p "$OUT"

for pair in "baseline:qwen3_8_27b_nvfp4.ninfer" "fold3:qwen3_8_27b_nvfp4_fold3.ninfer" \
            "foldmlp:qwen3_8_27b_nvfp4_foldmlp.ninfer"; do
  tag="${pair%%:*}"
  model="${pair#*:}"
  echo "===== $tag ($model) $(date +%H:%M:%S)"
  "$BIN" "/home/user/models/$model" --text "$CORPUS" > "$OUT/$tag.log" 2>&1
  rc=$?
  echo "rc=$rc"
  grep -iE 'perplexity|ppl|error' "$OUT/$tag.log" | tail -6
done
echo W8_PPL_DONE
