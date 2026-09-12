#!/bin/bash
# 把新 artifact 拷到原生盘（避免 9 分钟 drvfs 读），并核对大小
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
S=$J/data/dflash2_ref_head.ninfer
D=/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer
LOG=$J/dl/refhead_copy.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 拷贝新 artifact 到原生盘 $(date '+%H:%M:%S') ==="
ls -l "$S" | awk '{print "  src:", $5, $6, $7}'
if [ ! -f "$D" ] || [ "$(stat -c %s "$D")" != "$(stat -c %s "$S")" ]; then
  time cp -f "$S" "$D"
fi
ls -l "$D" | awk '{print "  dst:", $5, $6, $7}'
[ "$(stat -c %s "$D")" = "$(stat -c %s "$S")" ] && echo "  大小一致 ✓" || echo "  大小不一致 ✗"
df -h / | tail -1
echo COPY_REFHEAD_DONE
