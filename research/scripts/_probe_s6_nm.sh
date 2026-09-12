#!/usr/bin/env bash
# S6 probe: count actual kernel specializations in the pre-split objects (read-only nm).
set -u
O=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
for f in gqa_attention_decode_e8.cu.o gqa_attention_prefill.cu.o gqa_attention_prefill_e8.cu.o; do
  [ -f "$O/$f" ] || { echo "== $f: MISSING"; continue; }
  echo "== $f  ($(stat -c %s "$O/$f") bytes, $(stat -c %y "$O/$f" | cut -c1-19))"
done

echo
echo "=== decode_e8: distinct gqa_attention_decode_i8_tiled_kernel specializations ==="
timeout 300 nm -C --defined-only "$O/gqa_attention_decode_e8.cu.o" 2>/dev/null \
  | grep 'gqa_attention_decode_i8_tiled_kernel' \
  | sed -E 's/^[0-9a-f]+ +[A-Za-z] +//' | sort -u > /tmp/nm_e8.txt
echo "distinct kernel symbols: $(wc -l < /tmp/nm_e8.txt)"
for g in Gqa27Geometry Gqa35Geometry GqaMuseGeometry; do
  printf '  %-16s %s\n' "$g" "$(grep -c "$g" /tmp/nm_e8.txt)"
done
echo "  E8=false variants: $(grep -c 'false, true, Gqa' /tmp/nm_e8.txt || true)"
echo "  first 3 lines:"; sed -n '1,3p' /tmp/nm_e8.txt | cut -c1-260 | sed 's/^/    /'

echo
echo "=== prefill: distinct gqa_attention_prefill_i8_kernel specializations ==="
timeout 300 nm -C --defined-only "$O/gqa_attention_prefill.cu.o" 2>/dev/null \
  | grep 'gqa_attention_prefill_i8_kernel' \
  | sed -E 's/^[0-9a-f]+ +[A-Za-z] +//' | sort -u > /tmp/nm_p.txt
echo "distinct symbols: $(wc -l < /tmp/nm_p.txt)"
grep -o 'Gqa27Geometry\|Gqa35Geometry\|GqaMuseGeometry\|GqaPrefillBatchMetadata<true>\|GqaPrefillBatchMetadata<false>\|GqaPrefillDirectMetadata' /tmp/nm_p.txt \
  | sort | uniq -c | sed 's/^/    /'
echo "  E8 (i8_kernel<..., true>) count: $(grep -c 'true>$' /tmp/nm_p.txt || true)"
sed -n '1,2p' /tmp/nm_p.txt | cut -c1-240 | sed 's/^/    /'

echo
echo "=== prefill_e8 ==="
timeout 300 nm -C --defined-only "$O/gqa_attention_prefill_e8.cu.o" 2>/dev/null \
  | grep 'gqa_attention_prefill_i8_kernel' \
  | sed -E 's/^[0-9a-f]+ +[A-Za-z] +//' | sort -u > /tmp/nm_pe.txt
echo "distinct i8_kernel symbols: $(wc -l < /tmp/nm_pe.txt)"
sed -n '1,3p' /tmp/nm_pe.txt | cut -c1-240 | sed 's/^/    /'
echo
echo "=== prefill_e8: fill kernels ==="
timeout 300 nm -C --defined-only "$O/gqa_attention_prefill_e8.cu.o" 2>/dev/null \
  | grep -o 'gqa_attention_prefill_fill_i8[a-z_]*kernel' | sort | uniq -c | sed 's/^/    /'
