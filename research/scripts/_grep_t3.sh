#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== program_impl.h size ==="
wc -l "$R/src/targets/qwen3_6/impl/runtime/program_impl.h"
echo "=== dflash2 ingress lines in program_impl.h ==="
grep -n 'dflash2\|proposal_extents\|target_valid_columns\|DFlash2DecodeIngress\|licensed_counts\|accepted_drafts\|proposal_positions\|proposal_ids\|execution_frontiers\|context_frontiers\|anchors' \
  "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | head -80
echo
echo "=== grep source_column_offset everywhere ==="
grep -rn 'source_column_offset' "$R/src" | head -20
echo
echo "=== grep proposal_positions / proposal_ids everywhere (src) ==="
grep -rn 'proposal_positions\|proposal_ids\|target_argmax\|target_valid_columns' "$R/src" | head -60
