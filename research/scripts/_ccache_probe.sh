#!/bin/bash
# ccache 到底是"没生效"还是"生效但没命中"？用一个 5 行内核做受控实验（微秒级内存，不碰红线）。
export PATH="/home/user/.local/bin:$PATH"
D=/tmp/ccache_probe
rm -rf "$D"; mkdir -p "$D"; cd "$D" || exit 1

cat > tiny.cu <<'EOF'
#include <cuda_runtime.h>
__global__ void tiny_kernel(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { p[i] = p[i] * 2.0f + 1.0f; }
}
void tiny_launch(float* p, int n) { tiny_kernel<<<1, 32>>>(p, n); }
EOF

echo '=== 环境里的 ccache 变量 ==='
env | grep -i ccache || echo '  (无)'
echo
echo '=== 关键配置 ==='
ccache -p 2>&1 | grep -iE 'max_size|sloppiness|compiler_check|hard_link|inode_cache|run_second_cpp|depend_mode|absolute_paths|compression' | head -12
echo
echo '=== 实验：同一文件编两次（第二次应为 HIT）==='
before=$(ccache -s | grep -E 'Hits|Misses' | tr -d ' ')
echo "  之前: $before"
/usr/bin/time -f '  第一次耗时 %e s' ccache nvcc -c -O3 -std=c++20 \
    "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" -lineinfo \
    tiny.cu -o tiny1.o 2>&1 | tail -2
/usr/bin/time -f '  第二次耗时 %e s' ccache nvcc -c -O3 -std=c++20 \
    "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" -lineinfo \
    tiny.cu -o tiny2.o 2>&1 | tail -2
echo
echo '=== 之后 ==='
ccache -s | sed -n '1,12p'
echo
echo '=== 逐次判定（-v 能看到 HIT/MISS 原因）==='
ccache -v nvcc -c -O3 -std=c++20 "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]" \
    -lineinfo tiny.cu -o tiny3.o 2>&1 | grep -iE 'result|hit|miss|cache' | head -6
