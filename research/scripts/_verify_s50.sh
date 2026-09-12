#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
cd $R || exit 3
F="$J/_collab/E7_s50_kv_coverage.diff"
echo "=== [1] dry-run against the REAL build tree (fuzz=0, forward) ==="
patch -p1 --dry-run --forward --fuzz=0 < "$F" 2>&1 | tail -8
echo
echo "=== [2] independent check: the deleted half-predicate and the early return ==="
grep -n 'target < address.page_count\|target == address.page_count\|target <= address.page_count\|KV materialization exceeds' \
  "$R/src/targets/qwen3_6/impl/runtime/logical_kv_store.h" | head -6
echo
echo "=== [3] independent check: the reachable-shrink pair E7 cites ==="
sed -n '9913,9919p' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | cat -n | sed 's/^/   9913+/'
sed -n '11412,11418p' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | cat -n | sed 's/^/   11412+/'
grep -n 'ensure_sequence_kv_mapped\|materialize_sequence_kv' "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" | sed -n '1,3p;8,12p'
echo
echo "=== [4] how many call sites will be renamed ==="
grep -c 'materialize_to_tokens\|materialize_sequence_kv' "$R/src/targets/qwen3_6/impl/runtime/logical_kv_store.h" \
  "$R/src/targets/qwen3_6/impl/runtime/program.h" "$R/src/targets/qwen3_6/impl/runtime/program_impl.h" 2>/dev/null
echo
echo "=== [5] build state (must be before the variant TUs) ==="
grep -oE '^\[[ 0-9]+%\][^"]*' /tmp/pa_make_1.log | tail -2
for p in $(pgrep -f bin/nvcc); do echo "  nvcc $(ps -o etime= -p $p | tr -d ' ')"; break; done
