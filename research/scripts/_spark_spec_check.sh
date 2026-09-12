#!/bin/bash
R=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== the spark spec we just produced ==="
cat "$R/tools/archkit/specs/spark-x2.5-4b_spec.json"
echo
echo "=== does the spec carry these fields at all? ==="
for k in partial_rotary rope_theta rotary headwise gate_attn hidden_act gelu tie layer_types sliding; do
  printf "  %-16s %s\n" "$k" "$(grep -c "$k" "$R/tools/archkit/specs/spark-x2.5-4b_spec.json")"
done
echo
echo "=== what adapt.py inspects (its gap detectors) ==="
grep -nE "tier|'hook'|'new_op'|'post'|partial|rope|gate|act_fn|gelu|tied|sliding|window" "$R/tools/archkit/adapt.py" | head -40 | cut -c1-150
