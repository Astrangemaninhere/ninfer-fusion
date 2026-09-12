#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
echo "=== data/dflash2_ckpts/ ==="
ls -lt --time-style=+%m-%d_%H:%M "$J/data/dflash2_ckpts" 2>/dev/null | head -20
echo
echo "=== data/ 下的 .ninfer artifact ==="
ls -lt --time-style=+%m-%d_%H:%M "$J/data"/*.ninfer 2>/dev/null | head -12
echo
echo "=== /home/user/models/ 下的 artifact ==="
ls -lt --time-style=+%m-%d_%H:%M /home/user/models/*.ninfer 2>/dev/null | head -12
echo
echo "=== 转换脚本是否在权威树里 ==="
ls -1 $R/tools/convert/qwen3_8_27b/ 2>/dev/null | head
echo
echo "=== C_artifact_manifest.md 全文 ==="
cat "$J/_collab/C_artifact_manifest.md" 2>/dev/null | head -40
