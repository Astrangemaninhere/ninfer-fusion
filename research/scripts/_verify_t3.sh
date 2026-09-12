#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build
echo "=== report exists? ==="
ls -la "$Z/T3_dflash2_position_family.md"
wc -l -c "$Z/T3_dflash2_position_family.md"
echo
echo "=== artifacts ==="
ls -la "$Z"/T3_*
echo
echo "=== diff sanity: last 8 lines ==="
tail -8 "$Z/T3_df2_position_probe.diff"
echo
echo "=== diff sanity: @@ header (should end without CR) ==="
sed -n '3p' "$Z/T3_df2_position_probe.diff" | cat -A
echo
echo "=== final dry-run (rc must be 0) ==="
cd /home/user/ninfer-fusion && patch -p1 --dry-run < "$Z/T3_df2_position_probe.diff" && echo "PATCH_DRYRUN_RC=0"
md5sum src/targets/qwen3_6/impl/runtime/program_impl.h
