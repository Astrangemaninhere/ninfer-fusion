#!/bin/bash
A=/home/user/ninfer-fusion
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== [1] qwen3_6_27b config.h DFlashConfig (dspark) key lines:"
grep -n 'struct DFlashConfig\|full_only\|local_layers\|query_heads\|kv_heads\|head_dim\|local_capacity\|local_window\|mask_token\|block_drafts' "$A/src/targets/qwen3_6_27b/impl/config.h" | head -30
echo
echo "=== [2] same file DFlash2Config key lines (with struct anchor):"
awk 'NR>=116 && NR<=150 {printf "%d\t%s\n", NR, $0}' "$A/src/targets/qwen3_6_27b/impl/config.h"
echo
echo "=== [3] artifact config is_causal:"
for f in "$W/_dflash2_config.json" "$W/_dflash2_config2.json" "$W/tmp/dflash2-fp8-config.json"; do
  if [ -f "$f" ]; then printf "%s : " "$f"; grep -o '"is_causal": *[a-z]*' "$f" | head -2; printf "   layer_types: "; grep -o '"layer_types": *\[[^]]*\]' "$f" | head -1; printf "   sliding_window: "; grep -o '"sliding_window": *[0-9]*' "$f" | head -3; fi
done
echo
echo "=== [4] trainer attention (train_dflash2.py:183-198 raw):"
awk 'NR>=183 && NR<=198 {printf "%d\t%s\n", NR, $0}' "$W/train_dflash2.py"
echo
echo "=== [5] swa.h doc line numbers:"
grep -n 'symmetric non-causal\|every query row sees every live temporary query row\|registered windows\|abs(p_j-p_i)' "$A/include/ninfer/ops/swa.h"
echo
echo "=== [6] dflash_impl.h branch lines (321/329/394):"
grep -n 'if constexpr (Config::query_heads == 32\|if constexpr (Config::query_heads == 40\|const bool local_layer\|unsupported DFlash attention geometry' "$A/src/targets/qwen3_6/impl/runtime/dflash_impl.h"
echo
echo "=== [7] swa launcher kernel mapping (post-patch numbering differs; use pre-patch from git-less tree):"
grep -n 'swa_split_partial_kernel\|kSwaMaxCandidateSplit\|direct_context_limit' "$A/src/ops/launcher/swa.cu" | head
echo
echo "=== [8] dflash2_impl.h swa call site line:"
grep -n 'ops::swa(' "$A/src/targets/qwen3_6/impl/runtime/dflash2_impl.h"
