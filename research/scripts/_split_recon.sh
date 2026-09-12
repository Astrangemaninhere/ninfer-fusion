#!/bin/bash
# ① 修正后的 ccache 探针（必须用绝对路径的编译器，模拟 CMake 的真实调用）
# ② nvfp4_tma 这个 44 分钟 TU 的 CMake 组织与头文件规模
export PATH="/home/user/.local/bin:$PATH"
CV=/usr/local/cuda-13.3/bin/nvcc
D=/tmp/ccache_probe2
rm -rf "$D"; mkdir -p "$D"; cd "$D" || exit 1
cat > tiny.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void tiny_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { p[i] = p[i] * 2.0f + 1.0f; }
}
void tiny_launch(float* p, int n) { tiny_kernel<<<1, 32>>>(p, n); }
EOF
FLAGS=(-c -O3 -std=c++20 "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" -lineinfo)
echo '=== ccache 绝对路径探针 ==='
echo "--- 第 1 次（应 MISS，输出全部保留）---"
ccache "$CV" "${FLAGS[@]}" tiny.cu -o t1.o; echo "  rc=$?"
echo "--- 第 2 次（应 HIT）---"
ccache "$CV" "${FLAGS[@]}" tiny.cu -o t2.o; echo "  rc=$?"
echo '--- 统计 ---'
ccache -s | grep -E 'Cacheable|Hits|Misses|Errors|Cache size' | head -8
echo
echo '=== 若 HIT：缓存上限够不够（大 TU 的 .o 有多大）==='
ls -la /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_nvfp4_tma.dir/ops/linear/nvfp4/*.o 2>/dev/null | awk '{printf "  %8.1f MB  %s\n", $5/1048576, $NF}'
ls -la /home/user/ninfer-fusion/build/src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode*.o 2>/dev/null | awk '{printf "  %8.1f MB  %s\n", $5/1048576, $NF}'
echo
echo '=== nvfp4_tma 相关的重头文件行数 ==='
R=/home/user/ninfer-fusion
for f in src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh \
         src/ops/linear/nvfp4/nvfp4_config.h src/ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h; do
  p="$R/$f"; [ -f "$p" ] && printf "  %6d 行  %s\n" "$(wc -l < "$p")" "$f"
done
echo
echo '=== CMake: nvfp4_tma 与 ops 目标的源文件是怎么列的 ==='
grep -nE 'nvfp4_w4a4_tma|ninfer_nvfp4_tma|ninfer_ops|GLOB|add_library|target_sources' "$R/src/CMakeLists.txt" | head -30 | cut -c1-140
