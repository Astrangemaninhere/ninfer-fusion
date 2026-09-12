#!/usr/bin/env bash
# S6 probe 3: (a) do prefill.cu / prefill_e8.cu emit the SAME E8 i8 kernels?
#               (b) symbol types for the public entries (evidence for the link story)
set -u
O=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher

extract() {  # <obj> -> same key normalisation as _s6_nm_arms.py, E8=true only
  timeout 300 nm --defined-only "$1" 2>/dev/null \
    | grep -F 'gqa_attention_prefill_i8_kernel' | grep -F 'vPK13__nv_bfloat16' \
    | sed -E 's/.*(gqa_attention_prefill_i8_kernel)/\1/; s/vPK13__nv_bfloat16.*//' \
    | grep 'Lb1EEE' | sort -u
}
extract "$O/gqa_attention_prefill.cu.o"    > /tmp/s6_pref_e8.txt
extract "$O/gqa_attention_prefill_e8.cu.o" > /tmp/s6_pref8_e8.txt
echo "E8 i8_kernel specializations in prefill.cu.o   : $(wc -l < /tmp/s6_pref_e8.txt)"
echo "E8 i8_kernel specializations in prefill_e8.cu.o: $(wc -l < /tmp/s6_pref8_e8.txt)"
echo "identical sets? $(cmp -s /tmp/s6_pref_e8.txt /tmp/s6_pref8_e8.txt && echo YES || echo NO)"
echo "--- the shared specializations (first 3, wrapped) ---"
fold -w 150 /tmp/s6_pref_e8.txt | head -6

echo
echo "=== symbol types of the pre-split public entries ==="
for f in gqa_attention_decode_e8.cu.o gqa_attention_prefill.cu.o gqa_attention_prefill_e8.cu.o; do
  echo "--- $f"
  timeout 300 nm --defined-only "$O/$f" 2>/dev/null | grep -E ' T | W ' | grep -E \
    'gqa_attention_(decode_e8_launch|prompt_launch|prompt_attention_launch|kv_append_launch)' \
    | sed -E 's/^[0-9a-f]+ //' | sort -u | cut -c1-110 | sed 's/^/    /'
done

echo
echo "=== how many 'W' kernel symbols vs 'T' public symbols per object ==="
for f in gqa_attention_decode_e8.cu.o gqa_attention_prefill.cu.o gqa_attention_prefill_e8.cu.o; do
  printf '  %-34s W=%s T=%s\n' "$f" \
    "$(timeout 300 nm --defined-only "$O/$f" 2>/dev/null | awk '$2=="W"' | wc -l)" \
    "$(timeout 300 nm --defined-only "$O/$f" 2>/dev/null | awk '$2=="T"' | wc -l)"
done
