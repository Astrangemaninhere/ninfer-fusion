#!/bin/bash
# T2 part 4: leftover checks. READ-ONLY (patch --dry-run only).
T=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
cd "$T" || exit 3

echo "=== [A] declfix re-apply behaviour ==="
echo "--- forward dry-run, default (no -f):"
patch -p1 --dry-run < "$C/N3_loader_declfix.diff" 2>&1 | head -8; echo "rc=$?"
echo "--- forward dry-run, --force (worst case if someone forces):"
patch -p1 --dry-run --force < "$C/N3_loader_declfix.diff" 2>&1 | head -12
echo "--- what does the tree have at loader.h:137-160 now:"
sed -n '135,150p' src/ops/kernel/gqa_isoquant_row_scale_loader.h

echo
echo "=== [B] decoder_state.cpp markers (who wrote it at 21:54:40?) ==="
grep -n 'S28\|S24\|S30\|S36\|round 1\|round 2\|N3\|window_table\|budget' src/targets/qwen3_6/impl/state/decoder_state.cpp | head -30

echo
echo "=== [C] S3_rowscale_pool vs landed declfix: loader.h hunks ==="
echo "--- S3 loader.h hunk headers:"; awk '/^--- a\/src\/ops\/kernel\/gqa_isoquant_row_scale_loader.h/{f=1} f&&/^@@/{print} f&&/^--- a\//&&!/loader.h/{f=0}' "$C/build/S3_rowscale_pool.diff"
echo "--- N3_loader_declfix hunk headers:"; grep '^@@' "$C/N3_loader_declfix.diff"
echo "--- S3 loader.cu/loader.h region context (first 12 lines of hunk 2):"
awk '/^--- a\/src\/ops\/kernel\/gqa_isoquant_row_scale_loader.h/{f=1} f&&/^@@/{c++} c==2{print; if(++n>14) exit}' "$C/build/S3_rowscale_pool.diff"

echo
echo "=== [D] S3 vs E_gqa_isoquant_geometry_fail_loud: row_scale.cuh hunk ranges ==="
echo "--- S3 row_scale.cuh hunks:"; grep '^@@' "$C/build/S3_rowscale_pool.diff" | head -6
echo "--- E_gqa hunks:"; grep '^@@' "$C/E_gqa_isoquant_geometry_fail_loud.diff"
echo "--- E_gqa touches (files):"; grep -E '^\+\+\+ ' "$C/E_gqa_isoquant_geometry_fail_loud.diff"

echo
echo "=== [E] The three apps/cli patches: overlapping hunks? ==="
for p in N1_kvbitbudget.diff N2_coldwindow_apps.diff N3_apps_hunk.diff; do
  echo "--- $p"; grep -E '^(\-\-\-|\+\+\+|\@\@)' "$C/$p" | grep -E '^(\-\-\- a|\@\@)'
done

echo
echo "=== [F] N3_recalibrate vs N3_loader_declfix: does recalibrate expect declfix's moved text? ==="
grep -n 'kv_rowscale_sidecar_apply_from_env\|kv_calibrate_apply_at_load' "$C/N3_recalibrate.diff" | head
echo "--- N3_recalibrate new file list:"; grep -E '^\+\+\+ ' "$C/N3_recalibrate.diff"

echo
echo "=== [G] E9_s52 layouts_impl.h reverse offsets: which tree lines are there now ==="
sed -n '730,750p' src/targets/qwen3_6/impl/runtime/layouts_impl.h | head -25

echo
echo "=== [H] search for any run log after 21:45 mentioning draft-tokens ==="
grep -rls 'draft-tokens' /mnt/c/Users/User/Documents/ziqinzhang --include='*.log' --include='*.txt' --include='*.md' 2>/dev/null | head -20

echo "(end)"
