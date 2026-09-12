#!/bin/bash
R=/home/user/ninfer-fusion
cd "$R" || exit 1
echo '=== 双向 op 支持的几何（约束在哪）==='
grep -nE 'query_heads|kv_heads|head_dim|static_assert|group|kGqa' src/ops/launcher/bidirectional_gqa_attention.h 2>/dev/null | head -14 | cut -c1-130
echo '  --- kernel 侧 assert ---'
grep -nE 'static_assert|QHeads|KvHeads|HeadDim|Group' src/ops/kernel/bidirectional_gqa_attention.cuh 2>/dev/null | head -12 | cut -c1-130
echo
echo '=== dflash2 草稿层的几何常量 ==='
grep -nE 'query_heads|kv_heads|head_dim|Group|group' src/targets/qwen3_6/impl/runtime/dflash2_impl.h | head -12 | cut -c1-130
echo
echo '=== dflash2 的 swa 调用上下文 244..262 ==='
awk 'NR>=244 && NR<=262 {printf "%4d| %s\n", NR, $0}' src/targets/qwen3_6/impl/runtime/dflash2_impl.h | cut -c1-130
echo
echo '=== dspark 的 bidirectional 调用上下文 318..336 ==='
awk 'NR>=318 && NR<=336 {printf "%4d| %s\n", NR, $0}' src/targets/qwen3_6/impl/runtime/dflash_impl.h | cut -c1-130
