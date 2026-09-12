#!/bin/bash
# W1 probe: environment + EOL + contract discovery
cd /home/user/ninfer-fusion || exit 3
echo "=== CRLF check (count of CR lines):"
for f in src/targets/qwen3_6/impl/runtime/dflash2_impl.h \
         src/targets/qwen3_6/impl/runtime/dflash_impl.h \
         src/ops/kernel/bidirectional_gqa_attention.cuh \
         src/targets/qwen3_6/impl/runtime/program_impl.h; do
  printf "%s : " "$f"
  grep -c $'\r' "$f"
done
echo "=== line/file for hardcoded geometry:"
grep -n "kBidirectionalGqa" src/ops/kernel/bidirectional_gqa_attention.cuh | head -20
echo "=== locate contracts V1/V3 across plausible roots:"
for root in /home/user /mnt/c/Users/User/Documents/ziqinzhang; do
  find "$root" -maxdepth 5 \( -name "V1_vllm_contract.md" -o -name "V3_original_dflash_contract.md" -o -name "V2_llamacpp_contract.md" \) 2>/dev/null
done
echo "=== wrapper dir:"
ls src/ops/wrapper/ 2>/dev/null | grep -i -E "bidir|gqa|attn" 
echo "=== grep bidirectional usage across tree:"
grep -rn "bidirectional_gqa" src include apps tools --include=*.h --include=*.cuh --include=*.cpp --include=*.cu 2>/dev/null | grep -v "^src/ops/kernel/bidirectional_gqa_attention.cuh" | head -40
