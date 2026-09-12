#!/bin/bash
# 打补丁后的关键一跑：不设 VLLM_WSL2_ENABLE_PIN_MEMORY（全局不锁页）+ --swap-space 0
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8136
LOG=$J/dl/run_uva.log
exec > >(tee -a "$LOG") 2>&1
echo "=== UVA 解耦后的跑（无全局锁页 + swap-space 0） $(date '+%H:%M:%S') ==="
free -g | head -2

HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/q3nvfp4 \
    --speculative-config '{"method":"dflash","model":"/home/user/models/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --swap-space 0 \
    --num-gpu-blocks-override 256 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 --skip-mm-profiling \
    --gpu-memory-utilization 0.70 --max-model-len 512 --max-num-seqs 1 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_uva.out 2>&1 &
SRV=$!
echo "server pid=$SRV  (pin_memory 全局=关, swap-space=0)"

ready=0
for i in $(seq 1 170); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  READY at $((i*5))s ($(date '+%H:%M:%S'))"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务退出于 $((i*5))s"; break; fi
  [ $((i % 6)) -eq 0 ] && echo "  ... $((i*5))s used=$(free -g | awk 'NR==2{print $3}')G avail=$(free -g | awk 'NR==2{print $7}')G gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪 srv 尾 20 ==="; tail -20 $J/dl/srv_uva.out; echo "=== 内存 ==="; free -g | head -3; exit 2
fi
echo "--- 发请求 + 抓逐位置指标 ---"
PORT=$PORT $V /home/user/ask_live.py
echo "--- srv SpecDecoding ---"
grep -E 'SpecDecoding' $J/dl/srv_uva.out | tail -4
kill $SRV 2>/dev/null
echo UVA_RUN_DONE
