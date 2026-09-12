#!/bin/bash
# 一体化：起服务（HTTP 探活）→ 贪心请求 → 抓逐位置指标 → 收工
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8131
cd /home/user
echo "=== 一体化参考跑 $(date '+%H:%M:%S') ==="
free -g | head -2

HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/qwen3_8_27b_nvfp4_hf \
    --speculative-config '{"method":"dflash","model":"/home/user/models/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --gpu-memory-utilization 0.85 --max-model-len 2048 --max-num-seqs 8 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_f6.out 2>&1 &
SRV=$!
echo "server pid=$SRV"

echo "--- 用 HTTP 探活（最多 750s） ---"
ready=0
for i in $(seq 1 150); do
  sleep 5
  if curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200; then
    ready=1; echo "  ready at $((i*5))s ($(date '+%H:%M:%S'))"; break
  fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务进程退出于 $((i*5))s"; break; fi
  [ $((i % 12)) -eq 0 ] && echo "  ... $((i*5))s  mem_avail=$(free -g | awk 'NR==2{print $7}')G"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪，err 尾 20 ==="; tail -20 $J/dl/srv_f6.out; exit 2
fi

echo "--- 发贪心请求 + 抓指标 ---"
PORT=$PORT $V /home/user/ask_live.py
rc=$?
echo "ask rc=$rc"
echo "--- 服务日志里的 SpecDecoding ---"
grep -E 'SpecDecoding' $J/dl/srv_f6.out 2>/dev/null | tail -5
echo "--- 清理 ---"
kill $SRV 2>/dev/null
sleep 3
echo F6_DONE
