#!/bin/bash
# 关掉 multimodal 路径（视觉 profiling 是内存尖峰来源），文本-only 参考跑
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8133
LOG=$J/dl/run_f8.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 文本-only 参考跑（禁 MM） $(date '+%H:%M:%S') ==="
free -g | head -2

HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model $J/models/Qwen3.8-27B-NVFP4-RTX5090 \
    --speculative-config '{"method":"dflash","model":"'"$J"'/data/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 \
    --skip-mm-profiling \
    --gpu-memory-utilization 0.80 --max-model-len 2048 --max-num-seqs 4 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_f8.out 2>&1 &
SRV=$!
echo "server pid=$SRV"

ready=0
for i in $(seq 1 160); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  ready at $((i*5))s ($(date '+%H:%M:%S'))"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务退出于 $((i*5))s"; break; fi
  if grep -qE 'unrecognized arguments|unexpected keyword|error: ' $J/dl/srv_f8.out 2>/dev/null; then
    echo "  参数被拒（快速失败）"; break
  fi
  [ $((i % 12)) -eq 0 ] && echo "  ... $((i*5))s avail=$(free -g | awk 'NR==2{print $7}')G"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪，srv 尾 20 ==="; tail -20 $J/dl/srv_f8.out; exit 2
fi
echo "--- 发请求 ---"
PORT=$PORT $V /home/user/ask_live.py
grep -E 'SpecDecoding' $J/dl/srv_f8.out 2>/dev/null | tail -4
kill $SRV 2>/dev/null
echo F8_DONE
