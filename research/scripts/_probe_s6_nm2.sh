#!/usr/bin/env bash
# how many nm entries exist per (mangled) kernel specialization?
set -u
O=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
timeout 300 nm --defined-only "$O/gqa_attention_decode_e8.cu.o" 2>/dev/null > /tmp/nm_raw_mangled.txt
echo "nm lines total: $(wc -l < /tmp/nm_raw_mangled.txt)"
echo
echo "--- lines whose symbol is mangled AND carries the kernel (sample of 6) ---"
grep 'gqa_attention_decode_i8_tiled_kernel' /tmp/nm_raw_mangled.txt | grep 'vPK13__nv_bfloat16' | head -6 | cut -c1-150
echo
echo "--- symbol types present ---"
awk '/gqa_attention_decode_i8_tiled_kernel/ {print $2}' /tmp/nm_raw_mangled.txt | sort | uniq -c
echo
echo "--- distinct keys after cutting at vPK13__nv_bfloat16 ---"
grep 'gqa_attention_decode_i8_tiled_kernel' /tmp/nm_raw_mangled.txt | grep 'vPK13__nv_bfloat16' \
  | sed -E 's/.*gqa_attention_decode_i8_tiled_kernel/gqa_attention_decode_i8_tiled_kernel/; s/vPK13__nv_bfloat16.*//' \
  | sort -u > /tmp/nm_keys.txt
wc -l < /tmp/nm_keys.txt
echo
echo "--- per geometry (24/4/1/256 = G27, 16/2/2/256 = G35, 32/2/1/128 = Muse) ---"
for g in 24 4 1 256: :; do :; done
grep -c 'GqaGeometryILi24ELi4ELi1ELi256EE' /tmp/nm_keys.txt || true
grep -c 'GqaGeometryILi16ELi2ELi2ELi256EE' /tmp/nm_keys.txt || true
grep -c 'GqaGeometryILi32ELi2ELi1ELi128EE' /tmp/nm_keys.txt || true
echo
echo "--- per cache input ---"
grep -c 'GqaAppendInput' /tmp/nm_keys.txt || true
grep -c 'GqaCachedInput' /tmp/nm_keys.txt || true
echo
echo "--- one full key (mangled) for decoding ---"
head -1 /tmp/nm_keys.txt
echo
echo "--- E8 flag = 9th template arg: count keys with ...Lb1ENS0_14GqaAppendInputE ---"
grep -c 'Lb1ENS0_14GqaAppendInput' /tmp/nm_keys.txt || true
