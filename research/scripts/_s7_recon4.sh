set -u
R=/home/user/ninfer-fusion
S=/mnt/c/Users/User/Documents/ziqinzhang/_s7_scratch
mkdir -p $S
cp $R/src/ops/launcher/gqa_attention_decode.cu $S/decode_orig.cu
cp $R/src/ops/launcher/gqa_attention_decode_impl.cuh $S/impl.cuh
cp $R/src/ops/launcher/gqa_attention_decode_smallt.cu $S/smallt.cu
cp $R/src/ops/launcher/gqa_attention_decode_partial.cuh $S/partial.cuh
cp $R/src/ops/launcher/gqa_attention_decode_g35.cu $S/g35.cu
cp $R/src/ops/launcher/gqa_attention_decode_muse.cu $S/muse.cu
cp $R/src/ops/launcher/gqa_attention_decode_e8.cu $S/e8.cu
cp $R/src/ops/kernel/gqa_attention_geometry.cuh $S/geometry.cuh
cp $R/src/ops/softmax_attention/dense/causal_cache/small_t.cu $S/softmax_small_t.cu
cp $R/src/ops/launcher/gqa_attention.h $S/gqa_attention.h
echo copied
echo "=== references to g35/muse small-t entrypoints anywhere ==="
grep -rn 'gqa_attention_small_t_g35\|gqa_attention_cached_small_t_g35\|gqa_attention_small_t_muse\|gqa_attention_cached_small_t_muse' $R/src $R/tests $R/tools $R/apps 2>/dev/null | grep -v Binary
echo "=== references to small_t_launch_for anywhere ==="
grep -rn 'gqa_attention_small_t_launch_for' $R/src 2>/dev/null
echo "=== references to small_t_launch / cached_small_t_launch anywhere ==="
grep -rn 'gqa_attention_small_t_launch\b\|gqa_attention_cached_small_t_launch' $R/src 2>/dev/null | head -20
echo "=== geometry.cuh head ==="
sed -n '1,60p' $R/src/ops/kernel/gqa_attention_geometry.cuh
echo "=== decode_orig.cu: g35/muse/27 mentions ==="
grep -n 'G35\|g35\|Muse\|muse\|27Geometry' $R/src/ops/launcher/gqa_attention_decode.cu
echo "=== impl.cuh: which entry points / geometry use ==="
grep -n 'Gqa35Geometry\|GqaMuseGeometry\|Gqa27Geometry\|small_t_launch_for\|launch_for' $R/src/ops/launcher/gqa_attention_decode_impl.cuh | head -30
echo "=== decode_orig.cu function definitions (top-level) ==="
grep -n '^void \|^template\|^static\|^inline\|^\[\[noreturn\]\]\|^std::\|^bool \|^float \|^struct \|^namespace \|^#define' $R/src/ops/launcher/gqa_attention_decode.cu | head -60
echo "=== load of CUDA compile flags for impl-based TU ==="
head -3 $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_g35.cu.o.d
tail -3 $R/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode_g35.cu.o.d
