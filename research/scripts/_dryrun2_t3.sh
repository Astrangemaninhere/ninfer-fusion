#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build
tr -d '\r' < "$D/T3_mkprobe.py" > /tmp/mkprobe.py
python3 /tmp/mkprobe.py
echo "gen_rc=$?"
echo
cd /home/user/ninfer-fusion || exit 1
patch -p1 --dry-run < "$D/T3_df2_position_probe.diff"
echo "dry_rc=$?"
echo
echo "=== md5 unchanged after dry-run ==="
md5sum src/targets/qwen3_6/impl/runtime/program_impl.h
echo
echo "=== apply to a copy + bracket census ==="
rm -rf /tmp/t3apply && mkdir -p /tmp/t3apply/src/targets/qwen3_6/impl/runtime
cp src/targets/qwen3_6/impl/runtime/program_impl.h /tmp/t3apply/src/targets/qwen3_6/impl/runtime/
cd /tmp/t3apply && patch -p1 < "$D/T3_df2_position_probe.diff" && echo "apply_rc=0"
python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_check.py 2>/dev/null || true
