#!/bin/bash
cd /mnt/c/Users/User/Documents/ziqinzhang
echo "=== _collab/board 里 dflash2 与 ckpt/repo/下载 相关 ==="
grep -rniE 'dflash2' _collab/*.md _board_append*.md _TODO.md 2>/dev/null | grep -iE 'ckpt|checkpoint|repo|hugging|hf\.co|download|下载|step_|权重来源|W9' | head -20
echo
echo "=== 训练/转换工具 ==="
ls -1 /home/user/ninfer-fusion/tools 2>/dev/null | head -20
echo "--- 与 dflash2 转换有关的脚本 ---"
ls -1 /home/user/ninfer-fusion/tools 2>/dev/null | grep -iE 'dflash|df2|convert|export|head' | head -10
echo
echo "=== 本地训练输出目录（已知位置候选） ==="
for d in /home/user/dflash2_train /home/user/train_out /home/user/ckpt /home/user/runs /home/user/ninfer-train; do
  [ -d "$d" ] && echo "存在: $d" && ls -1t "$d" 2>/dev/null | head -6
done
