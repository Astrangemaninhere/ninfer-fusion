#!/bin/bash
# 等拷贝完成 → 自动跑草稿头 A/B（现役 vs incoai 自洽头）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
S=$J/data/dflash2_ref_head.ninfer
D=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
LOG=$J/dl/ab_chain.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 等拷贝 $(date '+%H:%M:%S') ==="
SSZ=$(stat -c %s "$S")
for i in $(seq 1 200); do
  DSZ=$(stat -c %s "$D" 2>/dev/null || echo 0)
  [ "$DSZ" = "$SSZ" ] && { echo "  拷贝完成 $(date '+%H:%M:%S')  size=$DSZ"; break; }
  sleep 5
done
DSZ=$(stat -c %s "$D" 2>/dev/null || echo 0)
if [ "$DSZ" != "$SSZ" ]; then echo "  拷贝未完成（$DSZ / $SSZ），中止"; exit 2; fi

echo "=== 启动 A/B $(date '+%H:%M:%S') ==="
df -h / | tail -1
bash /home/user/ab_refhead.sh
echo AB_CHAIN_DONE
