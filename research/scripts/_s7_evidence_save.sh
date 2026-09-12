set -u
W=/mnt/c/Users/User/Documents/ziqinzhang
E=$W/_collab/build/evidence
mkdir -p $E
cp /tmp/s7_digest.out "$E/S7_g35muse_instantiation_digest.txt"
python3 /tmp/s7_sm_inv.py > "$E/S7_causal_smallt_instantiation_digest.txt" 2>&1
python3 $W/_tu_timeline.py > "$E/S7_tu_timeline_2026-09-10_2145.txt" 2>&1
{
  echo "=== S7 evidence: raw nm family counts (defined T/t/W/w) ==="
  date '+%F %H:%M:%S'
  B=/home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher
  for o in gqa_attention_decode_g35.cu.o gqa_attention_decode_muse.cu.o gqa_attention_decode.cu.o; do
    echo "--- $o  size=$(stat -c%s $B/$o)  mtime=$(stat -c%y $B/$o)"
    nm --defined-only $B/$o | awk '$2=="T"||$2=="t"||$2=="W"||$2=="w"' | wc -l
    nm --defined-only -C $B/$o | grep -o '[A-Za-z_0-9]*kernel<' | sort | uniq -c | sort -rn
  done
  echo "--- causal small_t.cu.o size=$(stat -c%s /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/softmax_attention/dense/causal_cache/small_t.cu.o)"
  echo "=== guard/drift greps ==="
  R=/home/user/ninfer-fusion/src/ops
  echo "impl.cuh require_nvfp4_geometry_dim hits: $(grep -c require_nvfp4_geometry_dim $R/launcher/gqa_attention_decode_impl.cuh || true)"
  echo "smallt.cu require_nvfp4_geometry_dim hits: $(grep -c require_nvfp4_geometry_dim $R/launcher/gqa_attention_decode_smallt.cu || true)"
  grep -n 'static_assert(QKKs' $R/kernel/gqa_attention_decode_nvfp4.cuh
  grep -n 'reduce_grid(' $R/launcher/gqa_attention_decode_impl.cuh $R/launcher/gqa_attention_decode_smallt.cu $R/launcher/gqa_attention_decode.cu
  grep -n 'require_nvfp4_geometry_dim\|ft::observe' $R/launcher/gqa_attention_decode_partial.cuh | head
} > "$E/S7_raw_counts_and_greps.txt" 2>&1
ls -la $E/S7_*
