#!/bin/bash
# 最终尝试：KV=FP8（对齐模型的 kv_cache_quant_algo），模型与草稿都直接读 /mnt/c（不再占 WSL 盘）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8132
LOG=$J/dl/run_f7.log
exec > >(tee -a "$LOG") 2>&1
echo "=== KV=FP8 参考跑 $(date '+%H:%M:%S') ==="
free -g | head -2
df -h / | tail -1

HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model $J/models/Qwen3.8-27B-NVFP4-RTX5090 \
    --speculative-config '{"method":"dflash","model":"'"$J"'/data/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.80 --max-model-len 2048 --max-num-seqs 8 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_f7.out 2>&1 &
SRV=$!
echo "server pid=$SRV (KV=fp8)"

ready=0
for i in $(seq 1 160); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  ready at $((i*5))s ($(date '+%H:%M:%S'))"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务退出于 $((i*5))s"; break; fi
  [ $((i % 12)) -eq 0 ] && echo "  ... $((i*5))s avail=$(free -g | awk 'NR==2{print $7}')G gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪，srv 尾 25 ==="; tail -25 $J/dl/srv_f7.out; exit 2
fi

echo "--- 发请求 ---"
PORT=$PORT $V /home/user/ask_live.py
echo "ask rc=$?"
grep -E 'SpecDecoding' $J/dl/srv_f7.out 2>/dev/null | tail -4
kill $SRV 2>/dev/null
echo F7_DONE
