#!/bin/bash
# Muse + quantized KV: is the corruption specific to the quantized path?
# Evidence to explain (2026-09-10): with --kv-dtype nvfp4 the headdbg dump's
# SECOND pass (decode) shows L00 exploding to ~1e6 and L01..L51 all NaN, while
# the first pass (prefill) is sane at L00. Muse's default KV table is BF16 and
# §116c says the nvfp4 path for Muse was never exercised — so compare.
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
MUS=/home/user/models/muse_glimmer_30b_nvfp4.ninfer
PROMPT='1, 2, 3, 4, 5, 6,'

[ -x "$CLI" ] || { echo "FATAL: CLI missing $CLI"; echo MUSE_KV_COMPARE=FAIL; exit 2; }
[ -f "$MUS" ] || { echo "FATAL: model missing $MUS"; echo MUSE_KV_COMPARE=FAIL; exit 2; }

for kv in bf16 nvfp4; do
  out=/tmp/muse_kv_$kv.out
  err=/tmp/muse_kv_$kv.err
  NINFER_HEADDBG=1 timeout 600 "$CLI" "$MUS" --prompt "$PROMPT" --max-new 4 \
    --no-thinking --greedy --no-cuda-graph --kv-dtype "$kv" --max-context 4096 \
    > "$out" 2> "$err"
  rc=$?
  nan=$(grep -c 'NAN' "$err" || true)
  layers=$(grep -cE 'L[0-9]+_' "$err" || true)
  first_nan=$(grep -m1 -n 'NAN' "$err" | cut -c1-70)
  echo "[$kv] rc=$rc nan_lines=$nan headdbg_lines=$layers"
  echo "      first_nan_line: ${first_nan:-<none>}"
  echo -n "      text: "; head -c 100 "$out" | tr -d '\r' | tr '\n' ' '; echo
  # pass-2 L00 magnitude: the tell that decode (not prefill) is broken
  echo "      L00_attn occurrences: $(grep -c 'L00_attn' "$err" || true)  (2 = prefill+decode passes)"
  grep -m2 'L00_attn' "$err" | sed 's/^/        /' | cut -c1-90
done

b=$(grep -c 'NAN' /tmp/muse_kv_bf16.err 2>/dev/null || echo 0)
n=$(grep -c 'NAN' /tmp/muse_kv_nvfp4.err 2>/dev/null || echo 0)
echo "=== verdict ==="
echo "  bf16 nan_lines=$b   nvfp4 nan_lines=$n"
if [ "$b" -eq 0 ] && [ "$n" -gt 0 ]; then
  echo "MUSE_KV_COMPARE=PASS  (缺陷定界在 Muse + 量化 KV)"
elif [ "$b" -gt 0 ] && [ "$n" -gt 0 ]; then
  echo "MUSE_KV_COMPARE=REVIEW (bf16 也 NaN ⇒ 与 KV 量化无关, 问题更靠前)"
else
  echo "MUSE_KV_COMPARE=REVIEW (需要看上面的逐行输出判读)"
fi
echo MUSE_KV_COMPARE_DONE
