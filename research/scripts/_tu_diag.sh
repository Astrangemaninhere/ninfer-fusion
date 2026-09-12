#!/bin/bash
# 拆 TU 之前必须量清：① 一个 nvcc 到底吃多少内存、② split-compile 是否真在并行、
# ③ ccache 为什么从不命中、④ nvcc 有没有 --threads 这类"把多核并起来当一个用"的开关。
export PATH="/home/user/.local/bin:$PATH"
R=/home/user/ninfer-fusion

echo '===== ① 当前 nvcc 进程树与内存 (RSS KB) ====='
ps -eo pid,ppid,rss,etime,comm | grep -E 'nvcc|cicc|cudafe|ptxas|fatbinary|gcc|cc1plus' | grep -v grep
echo
echo '--- 合计 RSS ---'
ps -eo rss,comm | grep -E 'nvcc|cicc|cudafe|ptxas|fatbinary|cc1plus' | awk '{s+=$1} END {printf "  %.2f GB\n", s/1048576}'
echo
echo '===== ② ccache 版本与统计 ====='
which ccache; ccache --version | head -3
echo '--- ccache -s ---'
ccache -s 2>&1 | head -25
echo
echo '===== ③ nvcc 的并行相关开关 ====='
/usr/local/cuda-13.3/bin/nvcc --help 2>/dev/null | grep -A3 -E '^--(threads|split-compile|split-compile-extended|compress-mode)' | head -40
echo
echo '--- 编译期实际用的 prepend flags ---'
echo "  NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS:-(未设置)}"
echo
echo '===== ④ 目标 TU 的规模 ====='
for f in src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu \
         src/ops/launcher/gqa_attention_decode_e8.cu \
         src/ops/launcher/gqa_attention_decode.cu \
         src/targets/qwen3_6_27b/impl/variant.cpp \
         src/runtime/engine/engine.cpp; do
  p="$R/$f"
  [ -f "$p" ] && printf "  %6d 行  %8d B  %s\n" "$(wc -l < "$p")" "$(stat -c %s "$p")" "$f" || echo "  [缺] $f"
done
echo
echo '--- 这些 TU 里有多少个 kernel 启动/模板实例化线索 ---'
for f in src/ops/linear/nvfp4/nvfp4_w4a4_tma.cu src/ops/launcher/gqa_attention_decode_e8.cu src/ops/launcher/gqa_attention_decode.cu; do
  p="$R/$f"
  [ -f "$p" ] || continue
  n_inst=$(grep -cE 'template|instantiat|INSTANTIATE|__global__|<<<' "$p")
  n_inc=$(grep -cE '^#include' "$p")
  echo "  $f: 模板/内核线索 $n_inst 处, #include $n_inc 个"
done
