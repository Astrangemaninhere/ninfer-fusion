#!/bin/bash
# T2 tree-state ledger: reverse/forward dry-run for every pending patch. READ-ONLY.
TREE=/home/user/ninfer-fusion
C=/mnt/c/Users/User/Documents/ziqinzhang/_collab
OUT=/mnt/c/Users/User/Documents/ziqinzhang/_t2_evidence
mkdir -p "$OUT"
cd "$TREE" || { echo "TREE MISSING"; exit 3; }
echo "tree=$TREE  date=$(date -Is)"
echo "patch version: $(patch --version | head -1)"

PATCHES="
A5b_attention_valid_width.diff
E9_s52_dflash2_k_slice.diff
N1_kvbitbudget.diff
N2_coldwindow.diff
N2_coldwindow_apps.diff
N3_recalibrate.diff
N3_apps_hunk.diff
N3_loader_declfix.diff
A_s32_w13_p1.diff
build/D4_registry_datadriven.diff
build/S3_rowscale_pool.diff
E_s51_silu_extreme_negative_test.diff
E_gqa_isoquant_geometry_fail_loud.diff
A3_spark_gelu_mul.diff
A3_spark_head_geometry.diff
A3_spark_headwise_gate.diff
E7_s50_regression_sketch.diff
"

for p in $PATCHES; do
  f="$C/$p"
  tag=$(echo "$p" | tr '/' '_')
  echo "##################### $p #####################"
  if [ ! -f "$f" ]; then echo "MISSING PATCH FILE: $f"; continue; fi
  echo "--- bytes=$(wc -c < "$f") lines=$(wc -l < "$f") hunks=$(grep -c '^@@' "$f") ---"
  echo "--- targets ---"
  grep -nE '^(--- |\+\+\+ |diff |new file|deleted file|rename )' "$f" | head -80

  echo "--- [REV] patch -p1 --dry-run -R ---"
  patch -p1 --dry-run -R < "$f" > "$OUT/$tag.rev.txt" 2>&1
  rc=$?
  echo "REV rc=$rc"
  cat "$OUT/$tag.rev.txt"

  echo "--- [FWD] patch -p1 --dry-run ---"
  patch -p1 --dry-run < "$f" > "$OUT/$tag.fwd.txt" 2>&1
  rc2=$?
  echo "FWD rc=$rc2"
  cat "$OUT/$tag.fwd.txt"

  echo "--- [GIT-REV] git apply --check -R -p1 ---"
  git apply --check -R -p1 "$f" > "$OUT/$tag.gitrev.txt" 2>&1
  echo "GITREV rc=$?"
  cat "$OUT/$tag.gitrev.txt"

  echo "--- [GIT-FWD] git apply --check -p1 ---"
  git apply --check -p1 "$f" > "$OUT/$tag.gitfwd.txt" 2>&1
  echo "GITFWD rc=$?"
  cat "$OUT/$tag.gitfwd.txt"

  # residue check: did patch leave rej/orig anywhere in tree? (should never with --dry-run)
done
echo "DONE $(date -Is)"
