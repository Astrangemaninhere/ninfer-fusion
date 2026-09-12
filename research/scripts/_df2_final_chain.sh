#!/bin/bash
# 收尾链：对齐 flashinfer-cubin 版本（失败则用官方 bypass）→ 跑参考 → 打接受率
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
LOG=$J/dl/df2_final_chain.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 收尾链 $(date '+%F %H:%M:%S') ==="

echo "--- 1) 装同版本 flashinfer-cubin==0.6.18 ---"
$V -m pip install -i $IDX "flashinfer-cubin==0.6.18" 2>&1 | tail -4
$V -m pip list 2>/dev/null | grep -i flashinfer

echo "--- 2) 验证 flashinfer 可导入 ---"
BYPASS=""
if ! $V -c "import flashinfer; print('ok', flashinfer.__version__)" 2>&1 | tail -1 | grep -q '^ok'; then
  echo "  仍不可用，改用官方 bypass"
  BYPASS="FLASHINFER_DISABLE_VERSION_CHECK=1"
  FLASHINFER_DISABLE_VERSION_CHECK=1 $V -c "import flashinfer; print('bypass ok', flashinfer.__version__)" 2>&1 | tail -2
fi
echo "BYPASS='$BYPASS'"

echo "--- 3) 跑参考（原生盘路径 + pin_memory + enforce-eager） ---"
tr -d '\r' < $J/_vref_df2.py > /tmp/vd4.py
env $BYPASS setsid nohup $V /tmp/vd4.py > $J/dl/vref_final_stdout.log 2>&1 &
for i in $(seq 1 300); do
  sleep 5
  grep -qE 'VREF_DF2_DONE|ready=False' $J/dl/vref_final_stdout.log 2>/dev/null && break
  if grep -qE 'RuntimeError' $J/dl/vref_df2c.err 2>/dev/null && ! pgrep -f '/tmp/vd4.py' >/dev/null; then break; fi
done
echo "--- 结果 ---"
grep -E 'ready=|ACCEPTED=|DRAFTED=|RATE=|PER_POS=|SpecDecoding|text\[:60\]' $J/dl/vref_final_stdout.log 2>/dev/null | tail -12
echo "--- 若失败，根因 ---"
grep -nE 'RuntimeError|Error:|flashinfer|FlashInfer' $J/dl/vref_df2c.err 2>/dev/null | tail -5
echo DF2_FINAL_CHAIN_DONE
