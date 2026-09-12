#!/bin/bash
# T2 part 3: which OTHER patches (not in the T2 list) are already landed?
# Reverse dry-run probe of extra candidate patches. READ-ONLY.
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
cd "$T" || exit 3
for p in \
  E2_s44_acceptance_counter.diff \
  A_n1_patch.diff \
  A_n1b_cold_pages.diff \
  A_s32_w13_p0.diff \
  E3_s45c_spec_usage.diff \
  E3_s45d_prefill_guard_v2.diff \
  E3_s45b_s36_restore.diff \
  A_s24_window_table.diff \
  A_s30_budget_cold.diff \
  E3_s45_i8_plane_stride.diff \
  E3_s45_i8_plane_stride.diff \
  M_patchA_dflash2_state_slots.diff \
  A5_block_rows.diff \
  A5_round_probe.diff \
  E6_s48_dspark_verify_pos.diff \
  E4_s46_cold_policy_fix.diff \
  D_s35_frontload.diff \
  B_s33_ple_wiring.diff \
  B_s37_stage_a.diff \
  M_muse_pagefill_patch.diff \
  ; do
  f="$C/$p"
  [ -f "$f" ] || { echo "SKIP(no file)  $p"; continue; }
  out=$(patch -p1 --dry-run -R < "$f" 2>&1); rc=$?
  nfail=$(printf '%s' "$out" | grep -c 'ignored')
  printf 'REV rc=%d  ignored_hunks=%s   %s\n' "$rc" "$nfail" "$p"
  if [ "$rc" -eq 0 ]; then
    printf '      -> REVERSE APPLIES CLEANLY = ALREADY LANDED in tree\n'
    printf '%s\n' "$out" | sed 's/^/      | /' | head -8
  fi
done
echo "(end)"
