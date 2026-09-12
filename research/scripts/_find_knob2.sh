#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== draft_tokens 在 dflash2 运行时里的消费点 ==="
grep -rnE 'draft_tokens' $R/src/targets/qwen3_6_27b/ $R/src/targets/qwen3_6/impl/runtime/ 2>/dev/null | head -20
echo "=== 块构造：anchor + MASK×K ==="
grep -rnE 'mask_token|MASK|anchor' $R/src/targets/qwen3_6/impl/runtime/spec_decision.h 2>/dev/null | head -15
echo "=== dflash2 文档位置 ==="
ls -1 $R/docs/maintainer/ 2>/dev/null | head -20
