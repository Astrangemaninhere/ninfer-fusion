#!/bin/bash
# temp: acceptance battery for the W6 (bandwidth governor) + W16 P0 (lane failure) build.
#   0) snapshot the freshly built binary
#   1) W16 fault injection: request-domain fault must fail one request only
#   2) W16 invariant fault: engine must stop and /health must report failed
#   3) W6 bandwidth probe: decode latency under a concurrent 64K prefill, governor OFF vs ON
#   4) smoke regression (engine changes must not break the 7-step smoke)
set -u
ROOT=/mnt/c/Users/User/Documents/ziqinzhang

echo "########## 0/4 snapshot binary ##########"
mkdir -p /home/user/bin
SRC=/home/user/ninfer-fusion/build/apps/ninfer-serve
SNAP=/home/user/bin/ninfer-serve
cp -f "$SRC" "$SNAP" || { echo "SNAPSHOT_FAIL"; exit 1; }
a=$(md5sum "$SRC" | cut -d' ' -f1); b=$(md5sum "$SNAP" | cut -d' ' -f1)
if [ "$a" != "$b" ]; then echo "SNAPSHOT_MISMATCH"; exit 1; fi
echo "SNAPSHOT_OK $SNAP $a"
export NINFER_SERVE_BIN="$SNAP"

echo
echo "########## 1+2/4 W16 fault injection ##########"
bash "$ROOT/_w16_fault_test.sh" 2>&1 | tail -n 30

echo
echo "########## 3/4 W6 bandwidth governor (off vs on) ##########"
bash "$ROOT/_w6_bw_e2e.sh" 2>&1 | tail -n 45

echo
echo "########## 4/4 smoke regression ##########"
bash "$ROOT/smoke_full_nograph.sh" 2>&1 | tail -n 20

echo
echo "W6W16_BATTERY_DONE"
