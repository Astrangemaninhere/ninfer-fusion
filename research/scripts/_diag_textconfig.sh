#!/bin/bash
R=/home/user/ninfer-fusion
echo "=== TextConfig 从哪来（各 target 的 config.h 是独立还是共享基类） ==="
grep -rn 'using TextConfig\|struct TextConfig\|TextConfig =' "$R/src/targets/qwen3_6_27b/impl/config.h" "$R/src/targets/qwen3_6_35b_a3b/impl/config.h" "$R/src/targets/muse_glimmer_30b/impl/config.h" 2>/dev/null | head -6 | cut -c1-130
echo
echo "=== Muse 的两个成员长什么样（照抄形状） ==="
sed -n '38,45p;68,90p' "$R/src/targets/muse_glimmer_30b/impl/config.h" | cat -n | sed 's/^/  /' | cut -c1-130
echo
echo "=== 27b/35b config.h 里与窗口相关的现有内容 ==="
grep -nE 'sliding|window|full_attention_layers' "$R/src/targets/qwen3_6_27b/impl/config.h" | head -8 | cut -c1-130
echo
echo "=== layouts_impl.h 里 TextConfig 的定义处 ==="
grep -n 'TextConfig' "$R/src/targets/qwen3_6/impl/runtime/layouts_impl.h" | head -6 | cut -c1-130
