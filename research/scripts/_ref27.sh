#!/bin/bash
# 27GB 生效后：验证内存 + 重跑参考
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) 新内存是否生效 ==="
free -g | head -3
echo "=== 2) 关键资产 ==="
for p in /home/user/vllm029/bin/python /home/user/run_df2_ref.py /home/user/models/qwen3_8_27b_nvfp4_hf/config.json /home/user/models/draft_dflash2_ref/model.safetensors; do
  if [ -e "$p" ]; then echo "OK   $p"; else echo "MISS $p"; fi
done
echo "=== 3) 清掉可能残留的 server ==="
for p in $(pgrep -f 'vllm.entrypoints.openai.api_server' 2>/dev/null); do
  cl=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  case "$cl" in *vllm.entrypoints*) echo "  kill $p"; kill "$p" 2>/dev/null;; esac
done
sleep 2
echo "=== 4) 重跑（27GB VM 下） ==="
# 换日志名，避免和上次混淆
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path('/home/user/run_df2_ref.py')
t = p.read_text(encoding='utf-8', errors='replace')
t = t.replace('vref_f4.out', 'vref_f5.out').replace('vref_f4.err', 'vref_f5.err')
p.write_text(t)
print('ok', 'vref_f5.out' in t)
PYEOF
cd /home/user
HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup /home/user/vllm029/bin/python /home/user/run_df2_ref.py \
  > $J/dl/vref_f5_stdout.log 2>&1 &
echo "launched $(date '+%H:%M:%S')"
sleep 25
echo "--- 25 秒后 ---"
free -g | head -2
tail -3 $J/dl/vref_f5_stdout.log 2>/dev/null
pgrep -f 'run_df2_ref.py' >/dev/null && echo "runner 在跑 ✓" || echo "runner 不在 ✗"
