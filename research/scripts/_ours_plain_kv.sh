#!/bin/bash
# 对齐实验：我们引擎 plain，KV=fp8（对齐模型的 kv_cache_quant_algo=FP8），同协议同 prompt
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 3
echo "=== 对照：plain @ KV=fp8（对齐参考） ==="
./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --print-token-ids --kv-dtype fp8 \
  > $J/dl/ours_plain_fp8.log 2>&1
echo "rc=$?"
echo "--- 前 150 字符 ---"
grep -m1 -A2 'generated ids' $J/dl/ours_plain_fp8.log | head -3
echo "--- summary ---"
grep -E 'kv cache dtype|decode speed|generated tokens' $J/dl/ours_plain_fp8.log | head -4
echo
echo "=== 对照：plain @ KV=bf16（已知，作参照） ==="
./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 96 --max-context 4096 \
  --no-thinking --greedy --print-token-ids --kv-dtype bf16 \
  > $J/dl/ours_plain_bf16.log 2>&1
echo "rc=$?"
grep -E 'kv cache dtype|decode speed' $J/dl/ours_plain_bf16.log | head -3
echo OURS_PLAIN_DONE
