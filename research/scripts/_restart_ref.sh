#!/bin/bash
# WSL 崩后重启：清点 + 用持久路径（/home/user）重启参考跑
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== WSL 状态清点 $(date '+%F %H:%M:%S') ==="
echo "uptime: $(uptime -p 2>/dev/null)"
echo "--- 关键资产是否还在 ---"
for p in /home/user/vllm029/bin/python /home/user/models/qwen3_8_27b_nvfp4_hf/config.json /home/user/models/draft_dflash2_ref/model.safetensors /home/user/ninfer-fusion/build/apps/ninfer; do
  [ -e "$p" ] && echo "  OK   $p" || echo "  MISS $p"
done
echo "--- 崩溃前是否已起过 server（看日志最后状态） ---"
tail -6 $J/dl/vref_final_stdout.log 2>/dev/null
echo
echo "--- 重建持久脚本：runner ---"
cp -f $J/_vref_df2.py /home/user/run_df2_ref.py
# 修路径指向原生盘 + 换日志名
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path('/home/user/run_df2_ref.py')
t = p.read_text(encoding='utf-8', errors='replace')
t = t.replace('os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")', '"/home/user/models/qwen3_8_27b_nvfp4_hf"')
t = t.replace('os.path.join(J, "data", "draft_dflash2_ref")', '"/home/user/models/draft_dflash2_ref"')
t = t.replace('vref_df2.out', 'vref_f2.out').replace('vref_df2.err', 'vref_f2.err')
p.write_text(t)
print('paths ok:', '/home/user/models/qwen3_8_27b_nvfp4_hf' in t, '/home/user/models/draft_dflash2_ref' in t,
      'vref_f2.out' in t)
PYEOF
ls -l --time-style=+%H:%M /home/user/run_df2_ref.py

echo "--- 启动（持久路径 + 全部修复的 env） ---"
cd /home/user
HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup /home/user/vllm029/bin/python /home/user/run_df2_ref.py \
  > $J/dl/vref_f2_stdout.log 2>&1 &
echo "launched pid=$!"
sleep 20
tail -4 $J/dl/vref_f2_stdout.log 2>/dev/null
echo RESTART_LAUNCHED
