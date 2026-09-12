#!/bin/bash
# 打上 VLLM_WSL2_ENABLE_PIN_MEMORY=1 + --enforce-eager，然后重跑参考
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
cd "$J" || exit 3
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path('_vref_df2.py')
t = p.read_text(encoding='utf-8', errors='replace')
if 'VLLM_WSL2_ENABLE_PIN_MEMORY' not in t:
    t = t.replace('env["HF_HUB_OFFLINE"] = "1"',
                  'env["HF_HUB_OFFLINE"] = "1"\n'
                  'env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"   # WSL 默认 False -> UVA 不可用 (platforms/interface.py:1002)\n'
                  'env["VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"] = "0"')
if '"--enforce-eager"' not in t:
    t = t.replace('"--max-num-seqs", "16",', '"--max-num-seqs", "16",\n        "--enforce-eager",')
p.write_text(t)
print('patched pin_memory=%s enforce_eager=%s' % ('VLLM_WSL2_ENABLE_PIN_MEMORY' in t, '"--enforce-eager"' in t))
PYEOF
tr -d '\r' < _vref_df2.py > /tmp/vd2.py
pgrep -f '/tmp/vd.py' >/dev/null && pkill -f '/tmp/vd.py'
setsid nohup /home/user/vllm029/bin/python /tmp/vd2.py > $J/dl/vref_df2b_stdout.log 2>&1 &
echo relaunched
for i in $(seq 1 60); do
  sleep 5
  grep -qE 'VREF_DF2_DONE|ready=True|ready=False' $J/dl/vref_df2b_stdout.log 2>/dev/null && break
done
tail -12 $J/dl/vref_df2b_stdout.log
echo "--- err 关键行 ---"
grep -nE 'UVA|RuntimeError|ValueError|spec|SpecDecoding|startup complete' $J/dl/vref_df2.err 2>/dev/null | tail -8
