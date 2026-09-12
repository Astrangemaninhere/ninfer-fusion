set -u
R=/home/user/ninfer-fusion
B=$R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
echo "=== object sizes and mtimes ==="
ls -la --time-style=full-iso $B/gqa_attention_decode*.o 2>&1
echo "=== object sizes human-readable ==="
du -h $B/gqa_attention_decode_g35.cu.o $B/gqa_attention_decode_muse.cu.o $B/gqa_attention_decode.cu.o $B/gqa_attention_decode_e8.cu.o 2>&1
echo "=== is g35/muse in build.make? ==="
grep -n 'g35\|muse' $R/build/src/CMakeFiles/ninfer_ops.dir/build.make | head -20
echo "=== is g35/muse in link.txt? ==="
grep -c 'g35\|muse' $R/build/src/CMakeFiles/ninfer_ops.dir/link.txt 2>/dev/null || echo "no link.txt or 0"
echo "=== .o.d head for g35 ==="
head -5 $B/gqa_attention_decode_g35.cu.o.d 2>/dev/null
echo "=== nm: defined functions in g35.o (T/t/W/w) ==="
nm -C --defined-only $B/gqa_attention_decode_g35.cu.o 2>/dev/null | grep -c ' [TtWw] ' || nm -C --defined-only $B/gqa_attention_decode_g35.cu.o 2>&1 | head -5
echo "=== nm: total symbols g35 ==="
nm $B/gqa_attention_decode_g35.cu.o 2>/dev/null | wc -l
echo "=== nm: defined text symbols g35 sample ==="
nm -C --defined-only $B/gqa_attention_decode_g35.cu.o 2>/dev/null | grep ' [TtWw] ' | head -20
echo "=== nm muse ==="
nm $B/gqa_attention_decode_muse.cu.o 2>/dev/null | wc -l
nm -C --defined-only $B/gqa_attention_decode_muse.cu.o 2>/dev/null | grep ' [TtWw] ' | head -20
echo "=== which gcc/nm ==="
which nm cuobjdump 2>&1
