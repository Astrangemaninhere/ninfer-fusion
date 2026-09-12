#!/bin/bash
# Muse verification after the row-scale bounds fix (see _TODO.md 109/116):
#   [1] per-layer headdbg NaN sweep   [2] "1,2,3,4,5,6," -> 7 functional probe
#   [3] Chinese prompt sample
# Every step ASSERTS. A crashed CLI or a missing model must never read as "no NaN"
# (rc!=0 with zero headdbg lines looked identical to a clean run before).
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
MUS=/home/user/models/muse_glimmer_30b_nvfp4.ninfer
fail=0
note() { echo "[$1] $2"; }

[ -x "$CLI" ] || { echo "FATAL: CLI not executable: $CLI"; echo MUSE_VERIFY_FAIL; exit 2; }
[ -f "$MUS" ] || { echo "FATAL: model missing: $MUS"; echo MUSE_VERIFY_FAIL; exit 2; }

echo "=== [1] per-layer NaN check (nvfp4) ==="
NINFER_HEADDBG=1 timeout 300 "$CLI" "$MUS" \
  --prompt '1, 2, 3, 4, 5, 6,' --max-new 2 --no-thinking --greedy --no-cuda-graph \
  --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/mv1.out 2>/tmp/mv1.err
rc=$?
nan=$(grep -c 'NAN' /tmp/mv1.err || true)
layers=$(grep -cE 'L[0-9]+_' /tmp/mv1.err || true)
echo "rc=$rc nan_probe_lines=$nan headdbg_layer_lines=$layers"
grep -E 'L1[5-9]_(attn_out|attn|mlp)' /tmp/mv1.err | head -6
if [ "$rc" -ne 0 ] || [ "$nan" -ne 0 ] || [ "$layers" -lt 2 ]; then
  note nan FAIL "rc=$rc nan=$nan headdbg_lines=$layers (rc!=0 或 headdbg 无输出均判失败)"
  fail=1
else
  note nan PASS "nan=0 headdbg_lines=$layers"
fi

echo "=== [2] functional probe: 1..6 -> 7 ? ==="
timeout 300 "$CLI" "$MUS" --prompt '1, 2, 3, 4, 5, 6,' --max-new 8 --no-thinking --greedy \
  --no-cuda-graph --kv-dtype nvfp4 --max-context 4096 --print-token-ids > /tmp/mv2.out 2>/tmp/mv2.err
rc=$?
tail -6 /tmp/mv2.out
# 提示里没有 7, 所以输出中出现独立 "7" 就是续写成功的证据 (同时把 ids 行留档)。
if [ "$rc" -eq 0 ] && grep -q '7' /tmp/mv2.out; then
  note probe PASS "输出含 7; $(grep -oE 'generated ids.*' /tmp/mv2.err | head -1)"
else
  note probe FAIL "rc=$rc 未在输出中找到 7; $(grep -oE 'generated ids.*' /tmp/mv2.err | head -1)"
  fail=1
fi

echo "=== [3] chinese sample ==="
timeout 300 "$CLI" "$MUS" --prompt '用一句话介绍杭州。' --max-new 48 --no-thinking --greedy \
  --no-cuda-graph --kv-dtype nvfp4 --max-context 4096 > /tmp/mv3.out 2>/dev/null
rc=$?
tail -8 /tmp/mv3.out
bytes=$(wc -c < /tmp/mv3.out)
if [ "$rc" -eq 0 ] && [ "$bytes" -ge 30 ]; then
  note chinese PASS "rc=0 bytes=$bytes"
else
  note chinese FAIL "rc=$rc bytes=$bytes"
  fail=1
fi

if [ "$fail" -eq 0 ]; then echo MUSE_VERIFY_PASS; else echo MUSE_VERIFY_FAIL; fi
echo MUSE_VERIFY_DONE
