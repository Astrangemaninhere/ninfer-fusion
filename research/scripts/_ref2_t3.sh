#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== grep target-shift / target_shift (windows side) ==="
grep -rln 'target.shift\|target_shift\|TARGET_SHIFT' $Z/*.py $Z/data/df2pilot/*.py $Z/dl/*.py 2>/dev/null | head -20
echo
echo "=== which train_dflash2.py files exist and differ ==="
for f in $Z/train_dflash2.py $Z/data/df2pilot/train_dflash2.py $Z/dl/train_dflash2.py; do
  [ -f "$f" ] && echo "$f  $(wc -l < "$f") lines  md5=$(md5sum < "$f" | cut -c1-12)"
done
echo
echo "=== root train_dflash2.py: target/teacher lines ==="
grep -n 'target_shift\|ids16\|t16\|block =\|block_pos\|out\[:, 1:\]\|lm_head(out' $Z/train_dflash2.py 2>/dev/null | head -40
echo
echo "=== vllm-qwen3_dflash2.py size ==="
wc -l $Z/tmp/vllm-qwen3_dflash2.py $Z/_qwen3_dflash2.py $Z/_dflash2_speculator.py $Z/dl/aeon-dflash.py 2>/dev/null
echo
echo "=== dspark trainer: block/teacher lines ==="
grep -n 'block_ids\|teacher_src\|MASK\|mask\|teacher\[' $Z/train_dspark.py 2>/dev/null | head -40
