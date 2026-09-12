#!/usr/bin/env bash
# S6 probe 4: the E8 (true) i8 prefill-kernel set emitted by prefill.cu vs prefill_e8.cu
set -u
O=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
extract() {
  timeout 300 nm --defined-only "$1" 2>/dev/null \
    | grep -F 'gqa_attention_prefill_i8_kernel' | grep -F 'vPK13__nv_bfloat16' \
    | sed -E 's/.*(gqa_attention_prefill_i8_kernel)/\1/; s/vPK13__nv_bfloat16.*//' \
    | grep 'ELb1EEE$' | sort -u
}
extract "$O/gqa_attention_prefill.cu.o"    > /tmp/e8_a.txt
extract "$O/gqa_attention_prefill_e8.cu.o" > /tmp/e8_b.txt
printf 'prefill.cu.o    E8=true i8_kernel specializations: %s\n' "$(wc -l < /tmp/e8_a.txt)"
printf 'prefill_e8.cu.o E8=true i8_kernel specializations: %s\n' "$(wc -l < /tmp/e8_b.txt)"
if cmp -s /tmp/e8_a.txt /tmp/e8_b.txt; then echo "SAME SET (=> emitted in both TUs today, merged as weak symbols at link)"; else echo "DIFFERENT SETS"; fi
echo "--- first 2 keys (wrapped) ---"
fold -w 158 /tmp/e8_a.txt | head -4
