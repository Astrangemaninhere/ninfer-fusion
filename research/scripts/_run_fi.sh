#!/bin/bash
# 版本对齐 + fastsafetensors 加载器（不落主机）+ 最小形状；看门狗在宿主侧保护
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8138
LOG=$J/dl/run_fi.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 对齐版跑（fastsafetensors + 无 bypass） $(date '+%H:%M:%S') ==="
free -g | head -3

HF_HUB_OFFLINE=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/q3nvfp4 \
    --load-format fastsafetensors \
    --speculative-config '{"method":"dflash","model":"/home/user/models/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --num-gpu-blocks-override 256 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 --skip-mm-profiling \
    --gpu-memory-utilization 0.70 --max-model-len 512 --max-num-seqs 1 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_fi.out 2>&1 &
SRV=$!
echo "server pid=$SRV (fastsafetensors, flashinfer 0.6.13 对齐, 无 bypass)"

ready=0
for i in $(seq 1 170); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  READY at $((i*5))s ($(date '+%H:%M:%S'))"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务退出于 $((i*5))s"; break; fi
  if grep -qE 'unrecognized arguments|error: ' $J/dl/srv_fi.out 2>/dev/null; then echo "  参数被拒"; break; fi
  [ $((i % 6)) -eq 0 ] && echo "  ... $((i*5))s used=$(free -g | awk 'NR==2{print $3}')G avail=$(free -g | awk 'NR==2{print $7}')G gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪 srv 尾 25 ==="; tail -25 $J/dl/srv_fi.out; echo "=== 内存 ==="; free -g | head -3; exit 2
fi
echo "--- 发请求 ---"
PORT=$PORT $V /home/user/ask_live.py
echo "--- srv SpecDecoding ---"
grep -E 'SpecDecoding' $J/dl/srv_fi.out | tail -4
kill $SRV 2>/dev/null
echo FI_RUN_DONE
