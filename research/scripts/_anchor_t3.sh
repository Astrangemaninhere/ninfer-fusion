#!/bin/bash
R=/home/user/ninfer-fusion
D=/mnt/c/Users/User/Documents/ziqinzhang/_collab/build/T3_src
echo "=== program_impl.h includes (first 45 lines) ==="
sed -n '1,45p' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | tr -d '\r' | cat -n
echo
echo "=== does it use getenv/fprintf/cstdio already? ==="
grep -n '#include <cstdio>\|#include <cstdlib>\|std::getenv\|std::fprintf\|getenv(' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | head -10
echo
echo "=== anchor region 12432..12475 ==="
sed -n '12432,12475p' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | tr -d '\r' | cat -n
echo
echo "=== round_state.cpp: dflash decode tensor dims ==="
grep -n 'draft_tokens\|target_argmax\|proposal_ids\|proposal_positions\|accepted_drafts\|licensed_counts\|licensed_tokens' "$R/src/targets/qwen3_6/impl/state/round_state.cpp" | sed -n '1,40p'
