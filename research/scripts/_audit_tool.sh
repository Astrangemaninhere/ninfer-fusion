#!/bin/bash
echo "=== /home/user/dflash-src ==="
ls -la /home/user/dflash-src 2>&1 | head -20
echo
echo "=== 审计脚本接口 ==="
head -40 /mnt/c/Users/User/Documents/ziqinzhang/_collab/A5_weights_audit.py 2>&1
echo
echo "=== 转换脚本是否存在 ==="
ls -l /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo/tools/convert/qwen3_8_27b/patch_dflash2.py 2>&1
echo
echo "=== _collab/C_artifact_manifest.md 前 30 行（转换记录） ==="
head -30 /mnt/c/Users/User/Documents/ziqinzhang/_collab/C_artifact_manifest.md 2>&1
