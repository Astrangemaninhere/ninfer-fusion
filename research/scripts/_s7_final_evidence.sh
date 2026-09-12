set -u
R=/home/user/ninfer-fusion
W=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== binaries (relink gate) ==="
ls -la --time-style=long-iso $R/build/apps/ninfer $R/build/apps/ninfer-serve 2>/dev/null
find $R/build -maxdepth 2 -name 'ninfer*' -type f -newermt '2026-09-10 21:00' -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | head
echo "=== tests wiring for the two repro tests ==="
grep -rn 'repro_g35\|muse128_repro' $R/tests/CMakeLists.txt $R/CMakeLists.txt $R/src/CMakeLists.txt 2>/dev/null || echo "NOT in any CMakeLists -> not built"
echo "=== are the tests built anywhere in build tree? ==="
find $R/build -name '*repro_g35*' -o -name '*muse128*' 2>/dev/null | head
echo "=== evidence dir (naming convention) ==="
ls -la $W/_collab/build/evidence/ 2>/dev/null | head -20
echo "=== staged2 dir ==="
ls -la $W/_collab/build/staged2/ 2>/dev/null || echo "(staged2 does not exist yet)"
echo "=== live tree status of target files (final) ==="
for f in gqa_attention_decode.cu gqa_attention_decode_g35.cu gqa_attention_decode_muse.cu gqa_attention_decode_impl.cuh gqa_attention_decode_smallt.cu gqa_attention_decode_partial.cuh; do
  p=$R/src/ops/launcher/$f
  printf "%-42s %7d B  %5d lines  CR=%s  md5=%s\n" "$f" "$(stat -c%s $p)" "$(wc -l < $p)" "$(grep -c $'\r' $p || true)" "$(md5sum $p | cut -c1-32)"
done
p=$R/src/ops/softmax_attention/dense/causal_cache/small_t.cu
printf "%-42s %7d B  %5d lines  CR=%s  md5=%s\n" "causal/small_t.cu" "$(stat -c%s $p)" "$(wc -l < $p)" "$(grep -c $'\r' $p || true)" "$(md5sum $p | cut -c1-32)"
echo "=== CMakeLists line count + md5 + launcher block ==="
wc -l $R/src/CMakeLists.txt; md5sum $R/src/CMakeLists.txt
sed -n '64,80p' $R/src/CMakeLists.txt
echo "=== build.make refs to g35/muse (must be 0) ==="
grep -c 'g35\|muse' $R/build/src/CMakeFiles/ninfer_ops.dir/build.make || echo 0
echo "=== impl.cuh instantiation sites of launch_for (the 'thin wrapper' proof) ==="
grep -n 'gqa_attention_small_t_launch_for<' $R/src/ops/launcher/gqa_attention_decode_g35.cu $R/src/ops/launcher/gqa_attention_decode_muse.cu
echo "=== exported entities per file ==="
grep -n '^void \|^template\|^\[\[noreturn\]\]\|^bool \|^std::' $R/src/ops/launcher/gqa_attention_decode_g35.cu $R/src/ops/launcher/gqa_attention_decode_muse.cu
