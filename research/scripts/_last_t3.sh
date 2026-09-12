#!/bin/bash
Z=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== bat launchers ==="
ls $Z/*.bat 2>/dev/null | head -30
echo
echo "=== which script does each retrain/launch bat call ==="
grep -l 'train' $Z/*.bat 2>/dev/null | while read f; do
  echo "--- $(basename $f)"
  tr -d '\r' < "$f" | grep -iE 'python|train_dflash|target-shift|mask|df2pilot' | head -6
done
echo
echo "=== WSL models (dflash2/dspark artifacts) ==="
ls -la /home/user/models/ 2>/dev/null | grep -iE 'dflash|dspark' | head -20
echo
echo "=== is served df2 artifact == windows Aug26 copy? ==="
ls -la /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer 2>/dev/null
md5sum /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer 2>/dev/null
