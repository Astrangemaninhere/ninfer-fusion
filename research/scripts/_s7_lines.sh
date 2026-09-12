set -u
R=/home/user/ninfer-fusion
echo "=== geometry.cuh line numbers ==="
grep -n 'using Gqa\|struct GqaGeometry' $R/src/ops/kernel/gqa_attention_geometry.cuh
echo "=== nvfp4.cuh: D/QKKs/static_assert lines ==="
grep -n 'constexpr int D  *=\|constexpr int QKKs\|static_assert(QKKs' $R/src/ops/kernel/gqa_attention_decode_nvfp4.cuh
echo "=== decode.cu: small-t arms + fn lines (pristine) ==="
grep -n 'void gqa_attention_small_t_launch\|void gqa_attention_cached_small_t_launch\|Gqa35Geometry\|GqaMuseGeometry\|Gqa27Geometry' $R/src/ops/launcher/gqa_attention_decode.cu
echo "=== impl.cuh: launch_for + nvfp4 branch lines ==="
grep -n 'launch_tc_partial_nvfp4\|void gqa_attention_small_t_launch_for' $R/src/ops/launcher/gqa_attention_decode_impl.cuh
echo "=== smallt.cu: guard lines + launch_for ==="
grep -n 'require_nvfp4_geometry_dim\|if constexpr (Geometry::HeadDim\|launch_for<' $R/src/ops/launcher/gqa_attention_decode_smallt.cu
echo "=== build state: running procs ==="
ps -eo pid,etime,rss,comm,args --sort=-rss 2>/dev/null | grep -E 'ptxas|nvcc|make|cc1plus|c++|ld' | grep -v grep | head -6
echo "=== binaries/link outputs mtimes ==="
ls -la --time-style=long-iso $R/build/apps/ninfer $R/build/apps/ninfer-serve 2>/dev/null
find $R/build -maxdepth 3 -name '*.so' -o -maxdepth 3 -name 'ninfer-serve' -o -maxdepth 3 -name 'ninfer' 2>/dev/null | head
echo "=== newest build artifacts (top 6) ==="
find $R/build -newermt '2026-09-10 21:40' -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort -r | head -8
echo "=== staged2 md5s (other agent's, for non-collision record) ==="
md5sum /mnt/c/Users/User/Documents/ziqinzhang/_collab/build/staged2/* 2>/dev/null
