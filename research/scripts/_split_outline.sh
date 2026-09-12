#!/bin/bash
# 找拆分缝：不读全文，先出结构轮廓（函数/switch/实例化/宏）。
export PATH="/home/user/.local/bin:$PATH"
R=/home/user/ninfer-fusion
echo '=== ccache 上限（磁盘 287 GB 可用）==='
ccache --max-size=50G && ccache -p | grep -i max_size
ccache -s | grep -iE 'cache size' | head -2
echo
for f in src/ops/launcher/gqa_attention_decode.cu src/ops/launcher/gqa_attention_decode_e8.cu; do
  p="$R/$f"
  echo "===== $f ($(wc -l < "$p") 行) ====="
  echo '--- 顶层结构与控制流（缩进 0-2 的函数/switch/case/template/#define）---'
  grep -nE '^(#define|#include [<"])|^[A-Za-z_].*\(|^[[:space:]]{0,4}(switch|case |template|void |bool |int |static |using |struct |class |constexpr )' "$p" \
    | grep -vE '^\s*[0-9]+:#include' | head -60 | cut -c1-120
  echo
done
echo '=== ninfer_ops 的源列表开头（看新增文件该加在哪）==='
awk 'NR>=64 && NR<=100 {printf "%4d| %s\n", NR, $0}' "$R/src/CMakeLists.txt" | cut -c1-110
