#!/bin/bash
R=/home/user/ninfer-fusion
M=/mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 谁定义过 apply_from_env？（全树 + 镜像 + 协作区） ==="
grep -rn 'kv_rowscale_sidecar_apply_from_env' "$R/src" "$M/src" "$J/_collab" 2>/dev/null | head -8 | cut -c1-150
echo
echo "=== 头文件里现有的对外 API ==="
grep -nE '^[A-Za-z].*\(|^\s+[A-Za-z_].*\(.*\);|struct |class |install|parse|apply' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | grep -vE '^\s*//' | head -20 | cut -c1-140
echo
echo "=== 该头文件是否已经落进构建（被谁 include） ==="
grep -rln 'gqa_isoquant_row_scale_loader' "$R/src" 2>/dev/null | head
echo
echo "=== C_s28 协作目录里有什么（可能有原始实现与测试） ==="
ls -l "$J/_collab/C_s28_rowscale/" 2>/dev/null | head -12 | awk '{print "  ", $5, $NF}'
