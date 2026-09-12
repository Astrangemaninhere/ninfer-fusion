set -u
R=/home/user/ninfer-fusion/src/ops/launcher
echo "=== ft:: observe in live decode.cu / smallt.cu / impl.cuh ==="
grep -n 'ft::enabled\|ft::observe' $R/gqa_attention_decode.cu $R/gqa_attention_decode_smallt.cu $R/gqa_attention_decode_impl.cuh
echo "=== reduce_grid lines ==="
grep -n 'reduce_grid(' $R/gqa_attention_decode.cu $R/gqa_attention_decode_smallt.cu $R/gqa_attention_decode_impl.cuh
echo "=== live decode.cu small_t section header lines (for citation) ==="
sed -n '500,515p' $R/gqa_attention_decode.cu
echo "..."
sed -n '575,600p' $R/gqa_attention_decode.cu
echo "=== staged2 md5 record (other agent's files) ==="
md5sum /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged2/* 2>/dev/null | head -20
