set -u
R=/home/user/ninfer-fusion
B=$R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
echo "=== who includes impl.cuh / partial.cuh / smallt.cu ==="
grep -rn 'gqa_attention_decode_impl.cuh' $R/src $R/tests 2>/dev/null | grep -v Binary | head
echo "--- partial.cuh includers ---"
grep -rln 'gqa_attention_decode_partial.cuh' $R/src 2>/dev/null
echo "=== geometry definitions ==="
grep -rn 'struct Gqa27Geometry\|struct Gqa35Geometry\|struct GqaMuseGeometry\|Gqa35Geometry\b' $R/src --include=*.cuh --include=*.h --include=*.hpp 2>/dev/null | grep -i 'struct\|using\|=' | head -20
echo "=== g35/muse symbols in g35.o: grouped by kernel family ==="
nm -C $B/gqa_attention_decode_g35.cu.o 2>/dev/null | grep ' [Tt] ' | sed 's/^[0-9a-f]* [Tt] //' | sed 's/(.*//' | sed 's/\[clone.*//' | sort -u > /tmp/g35_syms.txt
wc -l /tmp/g35_syms.txt
cut -c1-100 /tmp/g35_syms.txt | sort | uniq -c | sort -rn | head -40
echo "=== muse symbols grouped ==="
nm -C $B/gqa_attention_decode_muse.cu.o 2>/dev/null | grep ' [Tt] ' | sed 's/^[0-9a-f]* [Tt] //' | sed 's/(.*//' | sed 's/\[clone.*//' | sort -u > /tmp/muse_syms.txt
wc -l /tmp/muse_syms.txt
cut -c1-100 /tmp/muse_syms.txt | sort | uniq -c | sort -rn | head -40
echo "=== distinct template arg lists (geometry instantiations) in g35.o ==="
grep -o 'GqaGeometry<[0-9, ]*>' /tmp/g35_syms.txt | sort -u
echo "=== distinct template arg lists in muse.o ==="
grep -o 'GqaGeometry<[0-9, ]*>' /tmp/muse_syms.txt | sort -u
echo "=== kernel template names in g35.o (unique) ==="
sed 's/<.*//' /tmp/g35_syms.txt | sort -u | head -30
echo "=== stubs only: __device_stub__ count ==="
nm $B/gqa_attention_decode_g35.cu.o 2>/dev/null | grep -c '__device_stub__'
nm $B/gqa_attention_decode_muse.cu.o 2>/dev/null | grep -c '__device_stub__'
echo "=== shared/dynsym text syms (global) in g35.o ==="
nm -C --defined-only $B/gqa_attention_decode_g35.cu.o 2>/dev/null | grep ' [TW] ' | head -20
echo "=== shared/dynsym text syms (global) in muse.o ==="
nm -C --defined-only $B/gqa_attention_decode_muse.cu.o 2>/dev/null | grep ' [TW] ' | head -20
