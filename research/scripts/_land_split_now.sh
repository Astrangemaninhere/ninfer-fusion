#!/bin/bash
# 落 gqa small-T + nvfp4 拆分：先移走树里的 stray 幽灵文件（内容=staged，已核），再预检，通过则落地
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
K=/home/user/kept_ghosts; mkdir -p "$K"
LOG=$J/dl/land_split.log
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo "=== land split (gqa small-T + nvfp4) $(date '+%F %H:%M:%S') ==="

# 0) 二次备份 stray 幽灵文件，然后移走（CMake 未引用 ⇒ 移走对构建零影响）
for f in src/ops/launcher/gqa_attention_decode_partial.cuh src/ops/launcher/gqa_attention_decode_smallt.cu; do
  if [ -f "$R/$f" ]; then
    echo "stray: $f md5=$(md5sum "$R/$f" | cut -d' ' -f1)"
    cp -f "$R/$f" "$K/$(basename "$f").$(date +%H%M)" || exit 3
    rm -f "$R/$f" || exit 3
    echo "  -> 已备份到 $K 并移走"
  fi
done
echo "staged 对照 md5:"
for n in gqa_attention_decode_partial.cuh gqa_attention_decode_smallt.cu; do
  echo "  $n staged=$(md5sum "$J/_collab/build/staged/$n.new" | cut -d' ' -f1)"
done

# 1) 预检
echo "--- preflight ---"
bash "$J/_land_split.sh" --dry-run > /tmp/ls_dry.log 2>&1
rc=$?
tail -6 /tmp/ls_dry.log
if [ "$rc" -ne 0 ]; then echo "PREFLIGHT_FAIL rc=$rc"; exit 2; fi

# 2) 落地
echo "--- land ---"
bash "$J/_land_split.sh" > /tmp/ls_land.log 2>&1
rc=$?
tail -18 /tmp/ls_land.log
[ "$rc" -ne 0 ] && { echo "LAND_FAIL rc=$rc"; exit 3; }

# 3) 回读关键事实
echo "--- 回读 ---"
wc -l $R/src/ops/launcher/gqa_attention_decode.cu $R/src/ops/launcher/gqa_attention_decode_partial.cuh \
      $R/src/ops/launcher/gqa_attention_decode_smallt.cu 2>/dev/null
grep -c 'gqa_attention_decode_partial.cuh' $R/src/ops/launcher/gqa_attention_decode.cu
grep -nE 'smallt.cu|nvfp4_w4a4_tma_(attn|gdn|mlp|residual)' $R/src/CMakeLists.txt | head -8
echo "--- 新产物里是否仍无 BF16 partial ---"
grep -cE 'static_cast<(const )?__nv_bfloat16\*>\s*\(\s*partial_acc\.data' $R/src/ops/launcher/gqa_attention_decode_partial.cuh
echo LAND_SPLIT_DONE
