#!/bin/bash
D=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) vLLM 相关日志与脚本 ==="
ls -lt "$D"/dl/vllm*.log "$D"/_vllm_*.py "$D"/*vllm*.bat 2>/dev/null | head -25
echo
echo "=== 2) 这些日志里的接受率/草稿配置 ==="
for f in "$D"/dl/vllm*.log; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$f") ---"
  grep -oiE 'draft[_a-z]*model[^,}]{0,60}|speculative[_a-z]*config[^,}]{0,80}|acceptance[_a-z ]*[0-9.]+%?|accept[_a-z ]*rate[^0-9]*[0-9.]+' "$f" 2>/dev/null | sort -u | head -8
done
echo
echo "=== 3) _vllm_*.py 里的草稿路径与参数 ==="
grep -nE 'draft|speculative|model=|MODEL|path' "$D"/_vllm_clean.py "$D"/_vllm_ref3.py 2>/dev/null | head -30
echo
echo "=== 4) 训练脚本 train_dflash2.py 的关键参数 ==="
find "$D" -name "train_dflash2.py" 2>/dev/null | head -3
grep -nE 'MASK_ID|mask_id|target_shift|mask_block|feature_layer|FEATURE_LAYERS|shift' $(find "$D" -name "train_dflash2.py" 2>/dev/null | head -1) 2>/dev/null | head -30
