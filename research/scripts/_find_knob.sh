#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== CLI 里与 draft 块尺寸/W 有关的旋钮 ==="
grep -nE '"--(draft|lm-head-draft|spec)' $R/apps/cli/options.cpp | head -20
echo "=== dflash2 block / W 的解析点 ==="
grep -rnE 'draft_tokens|draft_tokens_|block_size|mask_token|W *=' $R/apps/cli/options.cpp | head -20
echo "=== dflash2 契约里的 W/块定义（文档） ==="
grep -nE 'W *=|block|anchor|MASK' $R/docs/maintainer/qwen3.8-27b-dflash2.md 2>/dev/null | head -12
