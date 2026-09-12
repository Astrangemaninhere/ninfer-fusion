#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== UNIFY-A 补丁是否触及被拆文件 ==="
for f in launcher/gqa_attention_decode.cu launcher/gqa_attention_decode_e8.cu linear/nvfp4/nvfp4_w4a4_tma.cu; do
  n=$(grep -cE "^\+\+\+ b/src/ops/$f" $J/_collab/UNIFY_A_patch.diff 2>/dev/null)
  echo "  UNIFY-A hits src/ops/$f : $n"
done
echo "=== .split 内容是否已在树里出现（抽样判据） ==="
for s in "gqa_attention_uses_small_t" "gqa_attention_split_capacity" "gqa_small_t_launch_capacity"; do
  echo "  tree $(grep -c "$s" $R/src/ops/launcher/gqa_attention_decode.cu) / split $(grep -c "$s" $J/_collab/build/staged/gqa_attention_decode.cu.split)  <- $s"
done
echo "=== 树里 live TU 里被 PART-B 改的那 6 处现状 ==="
grep -cE 'static_cast<(const )?float\*>\s*\(\s*partial_acc\.data\s*\)' $R/src/ops/launcher/gqa_attention_decode.cu
grep -cE 'static_cast<(const )?__nv_bfloat16\*>\s*\(\s*partial_acc\.data\s*\)' $R/src/ops/launcher/gqa_attention_decode.cu
echo "=== 构建 ==="
pgrep -c nvcc
