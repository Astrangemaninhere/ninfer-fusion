#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 候选文件是否存在（定点检查，非扫盘） ==="
for f in \
  "$D/_collab/_dflash2_export_order_contract.md" \
  "$D/_collab/dflash2_export_order_contract.md" \
  "$D/train_dflash2.py" \
  "$D/ninfer-fusion-repo/tools/train/train_dflash2.py" \
  "$D/ninfer-fusion-repo/train_dflash2.py" \
  "$D/_collab/A2_draft_ceiling.md" ; do
  if [ -f "$f" ]; then echo "  [有] $f  ($(wc -l < "$f") 行)"; else echo "  [无] $f"; fi
done
echo
echo "=== A2 里关于训练输入构造的段落（抓关键行） ==="
grep -nE 'mask|shift|输入|input|block|target_layer|feature' "$D/_collab/A2_draft_ceiling.md" 2>/dev/null | head -25
