set -u
R=/home/user/ninfer-fusion
echo "=== digest for muse and main (filtered, no stub noise) ==="
grep -v '__device_stub__ZN' /tmp/s7_digest.out | sed -n '/===== muse/,$p'
echo ""
echo "=== softmax small_t object ==="
find $R/build -path '*softmax*small_t*' 2>/dev/null
echo "=== objects >20MB ==="
find $R/build -name '*.o' -size +20M -printf '%s %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort -rn | head -25
echo "=== nvfp4 kernel dir ==="
ls $R/src/ops/kernel/ | grep -i 'nvfp4\|gqa' | head
echo "=== QKKs static_assert ==="
grep -rn 'QKKs' $R/src/ops/kernel/gqa_attention_decode_nvfp4.cuh 2>/dev/null | head
echo "=== impl.cuh reduce grid HeadDim ==="
grep -n 'reduce_grid' $R/src/ops/launcher/gqa_attention_decode_impl.cuh
echo "=== impl.cuh ft::observe? ==="
grep -cn 'ft::observe' $R/src/ops/launcher/gqa_attention_decode_impl.cuh
echo "=== tests CMake refs ==="
grep -rn 'muse128_repro\|repro_g35' $R --include=CMakeLists.txt 2>/dev/null
ls $R/tests/*.cu 2>/dev/null | head -20
ls $R/build/tests 2>/dev/null | head
echo "=== land scripts on WSL side ==="
ls -la $R/*.sh $R/scripts/*.sh 2>/dev/null | head -20
ls /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged/
echo "=== _land_split.sh search (bounded) ==="
find /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 3 -name '*land*' -o -maxdepth 3 -name '*S4*' 2>/dev/null | head
