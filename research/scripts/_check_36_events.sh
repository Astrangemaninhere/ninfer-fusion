#!/bin/bash
echo "=== 生成第 36 个 token 附近，引擎日志里有什么事件？（中文 192 档，plain） ==="
grep -nE 'replan|capacity|grow|resize|materiali|页|page|re-?plan|reservation|graph|swap|defrag|compact' /home/user/op2_zh192_plain.log 2>/dev/null | head -20 | cut -c1-170
echo
echo "=== 该日志里所有非 load/summary 行（按时间顺序，前后各取） ==="
grep -vE '^\s*(load|summary|generate|tokens)\s' /home/user/op2_zh192_plain.log 2>/dev/null | head -30 | cut -c1-170
echo
echo "=== 中文 48 档（mtp）对比：同样位置有什么 ==="
grep -nE 'replan|capacity|grow|resize|reservation|graph' /home/user/op_zh192_mtp.log /home/user/op_m_mtp.log 2>/dev/null | head -12 | cut -c1-170
echo
echo "=== 引擎里"按生成计数触发"的机制（源码候选） ==="
R=/home/user/ninfer-fusion
grep -rnE 'kMaximumVerifyTokens|kMaximumBatchSize|kTwoChunkPromptVisibleKeys|kThreeChunkPromptVisibleKeys' "$R/src/ops/wrapper/gqa_attention.cpp" 2>/dev/null | head -6 | cut -c1-140
grep -rnE 'replan|regrow|grow_capacity|capacity_grow' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" 2>/dev/null | head -10 | cut -c1-150
