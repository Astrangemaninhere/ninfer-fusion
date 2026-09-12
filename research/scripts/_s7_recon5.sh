set -u
R=/home/user/ninfer-fusion
echo "=== digest for muse and main only (filtered) ==="
grep -v '__device_stub__ZN' /tmp/s7_digest.out | sed -n '/===== muse/,$p'
echo ""
echo "=== softmax small_t object ==="
find $R/build -path '*softmax*small_t*' 2>/dev/null
echo "=== all objects >20MB in build (top 30 by size) ==="
find $R/build -name '*.o' -size +20M 2>/dev/null -printf '%s %TY-%Tm-%Td %TH:%TM %p\n' | sort -rn | head -30
echo "=== nvfp4 static_assert evidence ==="
grep -n 'QKKs' $R/src/ops/kernel/gqa_attention_decode_nvfp4.cuh 2>/dev/null | head
ls $R/src/ops/kernel/ | grep -i nvfp4 | head
echo "=== impl.cuh HeadDim usage vs new ==="
grep -n 'HeadDim' $R/src/ops/launcher/gqa_attention_decode_impl.cuh | head -10
echo "--- smallt.cu (new) HeadDim usage ---"
grep -n 'HeadDim\|kGqaHeadDim' $R/src/ops/launcher/gqa_attention_decode_smallt.cu | head -10
echo "=== tests in build? ==="
grep -rn 'muse128_repro\|repro_g35' $R --include=CMakeLists.txt 2>/dev/null | head
ls -la $R/build/tests 2>/dev/null | head
echo "=== find _land_split.sh ==="
find / -maxdepth 6 -name '_land_split.sh' 2>/dev/null | head
find /mnt/c/Users/User/Documents/ziqinzhang -name '*land*' 2>/dev/null | head -20
echo "=== staged/ (previous round) contents ==="
ls -la /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged/ 2>/dev/null
echo "=== S4_split_landing.md head ==="
head -40 /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/S4_split_landing.md 2>/dev/null
