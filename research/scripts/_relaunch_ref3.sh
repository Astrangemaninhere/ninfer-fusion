#!/bin/bash
# 持久化重启参考跑：runner 写到 /home/user，全部 env 修复齐备
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 - <<'PYEOF'
import pathlib
src = pathlib.Path('/mnt/c/Users/User/Documents/ziqinzhang/_vref_df2.py')
t = src.read_text(encoding='utf-8', errors='replace')
t = t.replace('os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")', '"/home/user/models/qwen3_8_27b_nvfp4_hf"')
t = t.replace('os.path.join(J, "data", "draft_dflash2_ref")', '"/home/user/models/draft_dflash2_ref"')
t = t.replace('vref_df2.out', 'vref_f3.out').replace('vref_df2.err', 'vref_f3.err')
pathlib.Path('/home/user/run_df2_ref.py').write_text(t)
print('runner ready:', '/home/user/models/qwen3_8_27b_nvfp4_hf' in t, 'vref_f3.out' in t)
PYEOF
cd /home/user
HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup /home/user/vllm029/bin/python /home/user/run_df2_ref.py \
  > $J/dl/vref_f3_stdout.log 2>&1 &
echo "launched"
for i in $(seq 1 60); do
  sleep 5
  if grep -qE 'ready=True|ready=False|VREF_DF2_DONE' $J/dl/vref_f3_stdout.log 2>/dev/null; then break; fi
done
echo "--- 状态 ---"
tail -12 $J/dl/vref_f3_stdout.log 2>/dev/null
echo "--- err 关键行 ---"
grep -nE 'RuntimeError|Error:|FlashInfer|flashinfer|UVA|Traceback' $J/dl/vref_f3.err 2>/dev/null | tail -6
