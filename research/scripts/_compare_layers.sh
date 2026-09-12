#!/bin/bash
# 决定性对照：现树的 target_feature_layers vs 旧镜像 ninfer-fusion-repo 的值
NEW=/home/user/ninfer-fusion/src/targets/qwen3_6_27b/impl/config.h
OLD=/home/user/ninfer-fusion-repo/src/targets/qwen3_6_27b/impl/config.h
echo '=== 现树（WSL 权威树）==='
grep -n 'target_feature_layers' "$NEW" 2>/dev/null | cut -c1-140
echo
echo '=== 旧镜像（ninfer-fusion-repo，过期快照）==='
if [ -f "$OLD" ]; then
  grep -n 'target_feature_layers' "$OLD" 2>/dev/null | cut -c1-140
else
  echo "  镜像路径不存在，找找看:"
  find /home/user/ninfer-fusion-repo -name 'config.h' -path '*qwen3_6_27b*' 2>/dev/null | head -3
fi
echo
echo '=== 全盘搜其它可能的镜像/备份 ==='
for p in /home/user/ninfer-fusion-repo /mnt/c/Users/User/Documents/ziqinzhang/ninfer-fusion-repo /home/user/ninfer-fusion.orig; do
  [ -d "$p" ] && echo "  [在] $p" || echo "  [无] $p"
done
echo
echo '=== 现树里的 .orig 备份是否含旧值 ==='
find /home/user/ninfer-fusion/src -name 'config.h.orig' 2>/dev/null | head -5
for f in $(find /home/user/ninfer-fusion/src -name 'config.h.orig' 2>/dev/null | head -5); do
  echo "  --- $f ---"
  grep -n 'target_feature_layers' "$f" 2>/dev/null | cut -c1-130
done
echo
echo '=== 草稿训练脚本侧的层号（判官）==='
grep -n 'TARGET_LAYER_IDS\|target_layer_ids\|hs_layers' /mnt/c/Users/User/Documents/ziqinzhang/train_dflash2.py 2>/dev/null | head -6 | cut -c1-140
grep -n 'TARGET_LAYER_IDS\|target_layer_ids' /mnt/c/Users/User/Documents/ziqinzhang/train_dspark.py 2>/dev/null | head -4 | cut -c1-140
