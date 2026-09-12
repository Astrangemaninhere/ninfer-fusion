#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== 幽灵副本时间/大小 ==="
for f in src/ops/launcher/gqa_attention_decode_partial.cuh src/ops/launcher/gqa_attention_decode_smallt.cu; do
  if [ -f "$R/$f" ]; then
    stat -c '%y %s %n' "$R/$f"
  else
    echo "MISSING $f"
  fi
done
echo "=== 幽灵副本里的 partial_acc 声明 ==="
for f in src/ops/launcher/gqa_attention_decode_partial.cuh src/ops/launcher/gqa_attention_decode_smallt.cu; do
  echo "--- $f"
  grep -n 'partial_acc' "$R/$f" 2>/dev/null | head -8
done
echo "=== BF16 partial_acc 残留扫描（应为空） ==="
grep -rn '__nv_bfloat16' "$R/src/ops/launcher/gqa_attention_decode_partial.cuh" 2>/dev/null | grep -i partial | head
grep -cE 'bfloat16|bf16' "$R/src/ops/launcher/gqa_attention_decode_partial.cuh" 2>/dev/null
echo "=== 树里 live TU 的 partial_acc 类型（PART-B 已落） ==="
grep -n 'partial_acc' "$R/src/ops/launcher/gqa_attention_decode.cu" | head -5
echo "=== staged 产物 vs 树里幽灵副本 md5 对比（不等=staged 已过期） ==="
for n in gqa_attention_decode_partial.cuh gqa_attention_decode_smallt.cu; do
  echo "$n  tree=$(md5sum "$R/src/ops/launcher/$n" | cut -c1-12)  staged=$(md5sum "/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged/$n.new" | cut -c1-12)"
done
echo "=== 构建进度 ==="
tail -4 /mnt/c/Users/User/Documents/ziqinzhang/dl/land_part_b.log
