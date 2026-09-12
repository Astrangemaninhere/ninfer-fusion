#!/bin/bash
# 查"特征 tap"的既定语义（文档侧，与三路子代理读的 src 运行时代码不重叠）
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 权威文档里与 tap / feature 有关的内容 ==="
grep -rniE 'target_feature_layers|feature_rows|feature_layers|\btap\b|taps' $R/docs/maintainer/*.md 2>/dev/null | head -20
echo
echo "=== 2) 项目自己的报告/_collab 里提到 tap/特征的地方 ==="
grep -rniE 'target_feature_layers|feature_rows|\btap\b' $J/_collab/*.md 2>/dev/null | head -20
echo
echo "=== 3) 契约文档里 dflash2 的特征约定（若有） ==="
ls -1 $R/docs/maintainer/ | head -20
grep -rniE 'dflash2|draft' $R/docs/maintainer/qwen3.8-27b-artifact.md 2>/dev/null | head -20
echo
echo "=== 4) 源码里 tap 采集的注释（只读注释行，不碰实现） ==="
grep -rnE '^\s*//.*(tap|feature)' $R/src/targets/qwen3_6/impl/runtime/dflash_context.h $R/src/targets/qwen3_6/impl/runtime/dflash_context_impl.h 2>/dev/null | head -20
