#!/usr/bin/env bash
# S6 recon probe A: timings evidence + name-collision check (read-only)
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang

echo "=== par_build.log tail (CUDA object lines for gqa) ==="
if [ -f "$J/dl/par_build.log" ]; then
  grep -n 'gqa_attention' "$J/dl/par_build.log" | tail -12
  echo "--- log size/mtime ---"
  ls -la "$J/dl/par_build.log"
else
  echo "(no $J/dl/par_build.log)"
fi

echo
echo "=== any log mentioning the e8 TU ==="
grep -rl 'gqa_attention_decode_e8' "$J"/dl/*.log 2>/dev/null | head -5

echo
echo "=== name collision check (new symbol candidates) ==="
cd "$R" || exit 1
for n in \
  gqa_attention_decode_e8_append_g27 \
  gqa_attention_decode_e8_cached_g27 \
  gqa_attention_decode_e8_append_muse35 \
  gqa_attention_decode_e8_cached_muse35 \
  gqa_attention_decode_e8_arms \
  gqa_attention_prefill_arms \
  gqa_attention_prefill_e8_arms \
  gqa_attention_prefill_batch \
  gqa_attention_prefill_e8_kv_append \
  gqa_attention_prefill_e8_kv_append_single
do
  hits=$(grep -rl -- "$n" --include=*.cu --include=*.cuh --include=*.h --include=*.cpp src include tests 2>/dev/null | tr '\n' ' ')
  printf '%-46s %s\n' "$n" "${hits:-<none>}"
done
