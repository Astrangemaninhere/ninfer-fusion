set -u
R=/home/user/ninfer-fusion
echo "=== nvfp4 kernel: template decl + D binding + static_assert context ==="
sed -n '120,175p' $R/src/ops/kernel/gqa_attention_decode_nvfp4.cuh
echo ""
echo "=== kGqaKvQuantHeadDim / kGqaHeadDim definitions ==="
grep -rn 'kGqaKvQuantHeadDim\|constexpr int kGqaHeadDim' $R/src/ops/kernel/*.cuh $R/src/ops/launcher/*.h 2>/dev/null | head -10
echo "=== require_nvfp4_geometry_dim definition ==="
grep -n -A6 'require_nvfp4_geometry_dim' $R/src/ops/launcher/gqa_attention_decode.cu | head -20
echo "=== file mtimes: kernel headers vs Sep 9 ==="
ls -la --time-style=long-iso $R/src/ops/kernel/gqa_attention_decode_nvfp4.cuh $R/src/ops/kernel/gqa_attention_decode_i8.cuh $R/src/ops/kernel/gqa_attention_geometry.cuh $R/src/ops/launcher/gqa_attention_decode_impl.cuh 2>/dev/null
echo "=== softmax small_t.cu size/md5/CR ==="
ls -la --time-style=long-iso $R/src/ops/softmax_attention/dense/causal_cache/small_t.cu
md5sum $R/src/ops/softmax_attention/dense/causal_cache/small_t.cu
grep -c $'\r' $R/src/ops/softmax_attention/dense/causal_cache/small_t.cu || true
echo "=== softmax small_t object + fp8 object ==="
ls -la --time-style=long-iso $R/build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/small_t.cu.o $R/build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/small_t_fp8.cu.o 2>/dev/null
echo "=== softmax dir listing ==="
ls -la --time-style=long-iso $R/src/ops/softmax_attention/dense/causal_cache/ 2>/dev/null
echo "=== ccache stats (compile-count witness) ==="
ccache -s 2>/dev/null | head -20
echo "=== is the build still running? ==="
ps -eo pid,etime,rss,comm,args --sort=-rss 2>/dev/null | grep -E 'ptxas|nvcc|make|cc1plus' | grep -v grep | head -8
