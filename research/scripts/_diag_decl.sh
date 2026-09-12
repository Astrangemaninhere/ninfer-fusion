#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== .cu 里的完整定义（签名逐字抄） ==="
sed -n '1,40p' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.cu" | cat -n | sed 's/^/  /' | cut -c1-150
echo
echo "=== C_s28 的 patch 里是否本来有 .h 声明 ==="
grep -n 'apply_from_env' "$J/_collab/C_s28_rowscale/patch/C_s28_loader.patch" | head -6 | cut -c1-150
grep -n -B3 -A3 'apply_from_env' "$J/_collab/C_s28_rowscale/patch/C_s28_loader.patch" | grep -E '^\s*[0-9]+[-:]\s*\+.*(apply_from_env|\.h|bool )' | head -8 | cut -c1-160
echo
echo "=== 头文件里 parse 声明附近的上下文（决定插入点） ==="
sed -n '74,86p' "$R/src/ops/kernel/gqa_isoquant_row_scale_loader.h" | cat -n | sed 's/^/  74+/' | cut -c1-140
