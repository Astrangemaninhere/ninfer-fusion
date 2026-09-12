#!/bin/bash
echo '=== 找判官文件 ==='
for d in /home/user /mnt/c/Users/User/Documents/ziqinzhang /mnt/c/Users/User; do
  find "$d" -maxdepth 4 -name 'train_dspark*.py' -o -maxdepth 4 -name 'train_dflash2*.py' 2>/dev/null
done | sort -u | head -8
F=$(find /home/user /mnt/c/Users/User/Documents/ziqinzhang -maxdepth 4 -name 'train_dspark*.py' 2>/dev/null | head -1)
echo "  判官: ${F:-未找到}"
[ -n "$F" ] && {
  echo '=== block 构造与目标（关键行）==='
  grep -nE 'block|mask|ids16|target|shift|out\[|anchor|MASK_ID' "$F" | head -26 | cut -c1-135
}
echo
echo '=== 参考 runtime 的列↔位置（V1 引的 artifact 自带实现）==='
ls -la /home/user/dflash-src/dflash/model.py /home/user/ninfer-fusion/tmp/dspark/dflash.py /home/user/ninfer-fusion/tmp/vllm-qwen3_dflash2.py 2>/dev/null | cut -c25-100
grep -nE 'sample_off|start \+ j|start\+j|row .* predict|sample_from_anchor' /home/user/ninfer-fusion/tmp/dspark/dflash.py 2>/dev/null | head -8 | cut -c1-130
