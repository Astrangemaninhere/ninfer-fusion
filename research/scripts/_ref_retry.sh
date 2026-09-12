#!/bin/bash
# 精简重试：max-model-len 2048，只跑 zh 一个请求；观察崩在加载还是推理
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path('/home/user/run_df2_ref.py')
t = p.read_text(encoding='utf-8', errors='replace')
t = t.replace('"--max-model-len", "4096"', '"--max-model-len", "2048"')
t = t.replace('ask("hello", "hello")\n    ask(ZH, "zh")', 'ask(ZH, "zh")')
t = t.replace('vref_f2.out', 'vref_f4.out').replace('vref_f2.err', 'vref_f4.err')
p.write_text(t)
print('len2048:', '"2048"' in t, 'zh only:', 'ask("hello"' not in t)
PYEOF
cd /home/user
HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup /home/user/vllm029/bin/python /home/user/run_df2_ref.py \
  > $J/dl/vref_f4_stdout.log 2>&1 &
echo "launched $(date '+%H:%M:%S')"
for i in $(seq 1 100); do
  sleep 6
  if grep -qE 'VREF_DF2_DONE|ready=False' $J/dl/vref_f4_stdout.log 2>/dev/null; then break; fi
  if ! pgrep -f 'run_df2_ref.py' >/dev/null; then echo "runner 消失了（第 $((i*6))s）"; break; fi
done
echo "--- stdout ---"
tail -14 $J/dl/vref_f4_stdout.log 2>/dev/null
echo "--- err 尾 ---"
tail -6 $J/dl/vref_f4.err 2>/dev/null
