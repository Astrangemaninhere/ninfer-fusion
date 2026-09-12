#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 离线评测器与数据 ==="
ls -l --time-style=+%m-%d_%H:%M $J/_collab/A2_draft_eval.py 2>/dev/null
ls -l --time-style=+%m-%d_%H:%M $J/data/hs_cache_topk2/ 2>/dev/null | head -10
ls -l --time-style=+%m-%d_%H:%M $J/train_dflash2.py $J/_collab/train_dflash2.py 2>/dev/null
echo
echo "=== A2 报告里的评测命令与结论 ==="
grep -nE 'python|--ckpt|--hs|slot0|mk/eng|结论|hit@1|headroom|41.8|23,11,3,1' $J/_collab/A2_draft_ceiling.md 2>/dev/null | head -30
echo
echo "=== 磁盘余量（Windows 盘） ==="
df -h /mnt/c 2>/dev/null | tail -2
echo "=== 磁盘余量（WSL 侧） ==="
df -h /home 2>/dev/null | tail -2
