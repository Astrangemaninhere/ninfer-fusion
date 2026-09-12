set -u
R=/home/user/ninfer-fusion
echo "=== impl.cuh: guard present? ==="
grep -c 'require_nvfp4_geometry_dim' $R/src/ops/launcher/gqa_attention_decode_impl.cuh || true
grep -n 'if constexpr (Geometry::HeadDim' $R/src/ops/launcher/gqa_attention_decode_impl.cuh || echo "NO if-constexpr guard in impl.cuh"
echo "=== smallt.cu guard (for contrast) ==="
grep -c 'require_nvfp4_geometry_dim' $R/src/ops/launcher/gqa_attention_decode_smallt.cu || true
echo "=== dl/ and logs ==="
ls -la $R/dl/ 2>/dev/null | head -20
ls -la /mnt/c/Users/User/Documents/ziqinzhang/_collab/dl/ 2>/dev/null | head -20
echo "=== search logs for small_t timing / 171 ==="
for f in $R/dl/*.log /mnt/c/Users/User/Documents/ziqinzhang/_collab/dl/*.log; do
  [ -f "$f" ] && echo "--- $f" && grep -n 'small_t\|171\|elapsed\|real' "$f" 2>/dev/null | head -10
done
echo "=== window logs anywhere ==="
find /mnt/c/Users/User/Documents/ziqinzhang/_collab -maxdepth 2 -name '*window*' -o -maxdepth 2 -name '*.log' 2>/dev/null | head -20
find $R -maxdepth 2 -name 'window*.log' 2>/dev/null | head
echo "=== M_landed_set.md ==="
cat /mnt/c/Users/User/Documents/ziqinzhang/_collab/M_landed_set.md 2>/dev/null
echo "=== M_tree_hygiene.md ==="
cat /mnt/c/Users/User/Documents/ziqinzhang/_collab/M_tree_hygiene.md 2>/dev/null
