#!/bin/bash
# Muse bf16 NaN: bookkeeping (deterministic) vs race (uninitialised memory)?
# Implements discriminators 1 and 2 from A/S36's pivot (the SWA and KVHeads==2
# candidates were refuted with line evidence; the top candidate is a stale/
# unwritten cache slot exposed by position/frontier bookkeeping).
#
#   test 1 — repeat the SAME run twice at two generation lengths: if the failing
#            pass index is identical every time, the fault is deterministic
#            bookkeeping; if it moves, it is a race/uninitialised read.
#   test 2 — shift the prompt by one token: if the failing STEP index tracks the
#            absolute position, it is position bookkeeping.
set -u
CLI=/home/user/ninfer-fusion/build/apps/ninfer
MUS=/home/user/models/muse_glimmer_30b_nvfp4.ninfer

run_one() {  # tag prompt max-new
  local tag=$1 prompt=$2 maxnew=$3
  local err=/tmp/muse_disc_$tag.err out=/tmp/muse_disc_$tag.out
  NINFER_HEADDBG=1 timeout 600 "$CLI" "$MUS" --prompt "$prompt" --max-new "$maxnew" \
    --no-thinking --greedy --no-cuda-graph --kv-dtype bf16 --max-context 4096 \
    > "$out" 2> "$err"
  local rc=$?
  local nan=$(grep -c 'NAN' "$err" || true)
  # first pass index whose L00_attn is NaN (pass 0 = prefill)
  local firstbad
  firstbad=$(awk '
    /\[headdbg\] L00_attn/ {
      p++
      if ($0 ~ /nan/ && firstbad == "") firstbad = p - 1
    }
    END { print (firstbad == "" ? "none" : firstbad) }' "$err")
  printf '%-26s rc=%s nan=%-4s first_nan_pass=%s\n' "$tag" "$rc" "$nan" "$firstbad"
}

echo "=== test 1: repeatability at the same prompt (deterministic vs race) ==="
run_one "m1a" '1, 2, 3, 4, 5, 6,' 1
run_one "m1b" '1, 2, 3, 4, 5, 6,' 1
run_one "m4a" '1, 2, 3, 4, 5, 6,' 4
run_one "m4b" '1, 2, 3, 4, 5, 6,' 4

echo "=== test 2: prompt +1 token (does the failing step track position?) ==="
run_one "m7" '1, 2, 3, 4, 5, 6, 7,' 4
run_one "m8" '1, 2, 3, 4, 5, 6, 7, 8,' 4

echo "=== verdict ==="
a=$(grep -c NAN /tmp/muse_disc_m4a.err || true)
b=$(grep -c NAN /tmp/muse_disc_m4b.err || true)
if [ "$a" = "$b" ]; then
  echo "MUSE_DISC: same-length repeats give identical NaN counts ($a) -> deterministic (bookkeeping)"
else
  echo "MUSE_DISC: same-length repeats give DIFFERENT NaN counts ($a vs $b) -> race / uninitialised read"
fi
echo MUSE_DISC_DONE
