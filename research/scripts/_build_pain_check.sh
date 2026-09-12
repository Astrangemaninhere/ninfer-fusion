#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 当前编译进度（KV 门放宽那次）==="
tail -4 "$J/dl/kvdump_gate_build.log" 2>/dev/null
pgrep -a -x make | head -2
ls -l --time-style=+%H:%M "$R/build/apps/ninfer" 2>/dev/null
echo
echo "=== 暂存的 TU 拆分（提速解药）是否还在 ==="
ls -d "$J/_collab/build/staged" "$J/_collab/build/staged2" 2>/dev/null
ls "$J/_collab/build/staged/" 2>/dev/null | head -8
echo "--- 拆分落地脚本 ---"
ls -l "$J"/_land_split.sh "$J/_collab/build/"_land_split.sh 2>/dev/null | head -3
echo
echo "=== 巨型 TU 现状（重编代价来源）==="
for f in "$R/build/src/CMakeFiles/ninfer_engine.dir/targets/qwen3_6_27b/impl/variant.cpp.o" \
         "$R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o"; do
  [ -f "$f" ] && ls -l --time-style=+%H:%M "$f" | awk '{print "  ", $5, $6, $7}'
done
echo
echo "=== 依赖跟踪是否生效（看编译命令里有没有 -MD/-MMD）==="
grep -m2 -oE '\-MD|\-MMD|\-MF [^ ]+' "$R/build/src/CMakeFiles/ninfer_engine.dir/flags.make" 2>/dev/null | head -4
