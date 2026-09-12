#!/usr/bin/env bash
# S6 landing-script self-test in a sandbox copy (never touches the live tree).
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
T=/tmp/s6_test
SNAP=/tmp/s6_landdir

rm -rf "$T" "$SNAP"
mkdir -p "$T/src/ops/launcher" "$SNAP"

# sandbox == the relevant slice of the real tree
cp -p "$R/src/CMakeLists.txt" "$T/src/CMakeLists.txt"
for f in gqa_attention_decode_e8.cu gqa_attention_prefill.cu gqa_attention_prefill_e8.cu; do
  cp -p "$R/src/ops/launcher/$f" "$T/src/ops/launcher/$f"
done
echo "--- sandbox ---"
md5sum "$T/src/ops/launcher/"*.cu "$T/src/CMakeLists.txt"

echo
echo "############ (1) DRY-RUN ############"
LAND_TEST_ROOT=$T LAND_SNAPSHOT_DIR=$SNAP bash <(tr -d '\r' < "$J/_land_s6.sh") --dry-run
echo "dry-run rc=$?"

echo
echo "############ (2) LAND ############"
LAND_TEST_ROOT=$T LAND_SNAPSHOT_DIR=$SNAP bash <(tr -d '\r' < "$J/_land_s6.sh")
echo "land rc=$?"

echo
echo "############ (3) LAND AGAIN (idempotency) ############"
LAND_TEST_ROOT=$T LAND_SNAPSHOT_DIR=$SNAP bash <(tr -d '\r' < "$J/_land_s6.sh") | tail -5
echo "reland rc=$?"

echo
echo "############ (4) REVERT ############"
LAND_TEST_ROOT=$T LAND_SNAPSHOT_DIR=$SNAP bash <(tr -d '\r' < "$J/_land_s6.sh") --revert
echo "revert rc=$?"

echo
echo "############ (5) SANDBOX AFTER REVERT (must equal the pristine slice) ############"
ls -la "$T/src/ops/launcher/"
md5sum "$T/src/ops/launcher/"*.cu "$T/src/CMakeLists.txt"
echo "live-tree md5 for comparison:"
md5sum "$R/src/ops/launcher/gqa_attention_decode_e8.cu" "$R/src/ops/launcher/gqa_attention_prefill.cu" \
       "$R/src/ops/launcher/gqa_attention_prefill_e8.cu" "$R/src/CMakeLists.txt"
