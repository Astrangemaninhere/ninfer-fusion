#!/bin/bash
# 全自动链：等 prep 完成 → 用原生盘路径重跑参考 → FlashInfer 失败则自动换 attention backend 再试
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
LOG=$J/dl/df2_ref_chain.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 参考链 $(date '+%F %H:%M:%S') ==="

for i in $(seq 1 240); do
  grep -q 'DF2_PREP_DONE' $J/dl/df2_prep.log 2>/dev/null && break
  sleep 5
done
echo "prep: $(grep -c DF2_PREP_DONE $J/dl/df2_prep.log 2>/dev/null) 完成标记"
grep -iE 'flashinfer' $J/dl/df2_prep.log | tail -5
du -sh /home/user/models/qwen3_8_27b_nvfp4_hf /home/user/models/draft_dflash2_ref 2>/dev/null

# 把 runner 指向原生盘 + 换新日志名
python3 - <<'PYEOF'
import pathlib
p = pathlib.Path('/mnt/c/Users/User/Documents/ziqinzhang/_vref_df2.py')
t = p.read_text(encoding='utf-8', errors='replace')
t = t.replace('os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")', '"/home/user/models/qwen3_8_27b_nvfp4_hf"')
t = t.replace('os.path.join(J, "data", "draft_dflash2_ref")', '"/home/user/models/draft_dflash2_ref"')
t = t.replace('vref_df2.out', 'vref_df2c.out').replace('vref_df2.err', 'vref_df2c.err')
p.write_text(t)
print('runner 已指向原生盘:', '/home/user/models/qwen3_8_27b_nvfp4_hf' in t, '/home/user/models/draft_dflash2_ref' in t)
PYEOF
tr -d '\r' < $J/_vref_df2.py > /tmp/vd3.py

run_once() {  # $1 = tag, $2 = extra env (k=v;...)
  local tag=$1 extra=${2:-}
  echo "--- 尝试 $tag (extra env: ${extra:-none}) ---"
  env $extra setsid nohup $V /tmp/vd3.py > $J/dl/vref_${tag}_stdout.log 2>&1 &
  for i in $(seq 1 300); do
    sleep 5
    grep -qE 'VREF_DF2_DONE|ready=False' $J/dl/vref_${tag}_stdout.log 2>/dev/null && break
    grep -q 'FlashInfer backend is not available' $J/dl/vref_df2c.err 2>/dev/null && break
  done
  tail -8 $J/dl/vref_${tag}_stdout.log
  grep -qE 'ACCEPTED=' $J/dl/vref_${tag}_stdout.log 2>/dev/null && echo "SUCCESS_$tag"
}

run_once run1 ""
if ! grep -qE 'ACCEPTED=' $J/dl/vref_run1_stdout.log 2>/dev/null; then
  pkill -f '/tmp/vd3.py' 2>/dev/null
  echo "--- run1 未出数，换 attention backend 再试 ---"
  run_once run2 "VLLM_ATTENTION_BACKEND=TRITON_ATTN"
fi
echo "=== 汇总 ==="
for t in run1 run2; do
  echo "[$t]"; grep -E 'ACCEPTED=|ready=|SpecDecoding|RATE=' $J/dl/vref_${t}_stdout.log 2>/dev/null | tail -6
done
echo DF2_REF_CHAIN_DONE
