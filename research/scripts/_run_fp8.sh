#!/bin/bash
# 继续跑 FP8：拆掉 OOM 三成因（原生盘 + 27GB VM + 丢页缓存）后重跑
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8134
LOG=$J/dl/run_fp8.log
exec > >(tee -a "$LOG") 2>&1
echo "=== FP8 重跑（原生盘 + 丢缓存） $(date '+%H:%M:%S') ==="
free -g | head -3
df -h / | tail -1

echo "--- 1) 模型拷回原生盘（去 9P 的页缓存重复） ---"
mkdir -p /home/user/models/q3nvfp4
if [ ! -f /home/user/models/q3nvfp4/config.json ]; then
  cp -f $J/models/Qwen3.8-27B-NVFP4-RTX5090/* /home/user/models/q3nvfp4/ && echo "  拷完"
fi
du -sh /home/user/models/q3nvfp4 2>/dev/null
echo "--- 2) 丢页缓存（把刚拷入的页从 cache 里放掉） ---"
sync
(echo 3 > /proc/sys/vm/drop_caches) 2>/dev/null && echo "  drop_caches ok" || echo "  drop_caches 无权限（不影响正确性）"
free -g | head -2

echo "--- 3) 起服务（KV=fp8，文本-only） ---"
HF_HUB_OFFLINE=1 VLLM_WSL2_ENABLE_PIN_MEMORY=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/q3nvfp4 \
    --speculative-config '{"method":"dflash","model":"/home/user/models/draft_dflash2_ref","num_speculative_tokens":7,"draft_sample_method":"greedy"}' \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 --skip-mm-profiling \
    --gpu-memory-utilization 0.80 --max-model-len 2048 --max-num-seqs 4 \
    --enforce-eager --port $PORT --host 127.0.0.1 \
    > $J/dl/srv_fp8.out 2>&1 &
SRV=$!
echo "server pid=$SRV"

ready=0
for i in $(seq 1 170); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  READY at $((i*5))s ($(date '+%H:%M:%S'))"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "  服务退出于 $((i*5))s"; break; fi
  [ $((i % 12)) -eq 0 ] && echo "  ... $((i*5))s avail=$(free -g | awk 'NR==2{print $7}')G used=$(free -g | awk 'NR==2{print $3}')G"
done
if [ "$ready" != "1" ]; then
  echo "=== 未就绪 srv 尾 25 ==="; tail -25 $J/dl/srv_fp8.out; exit 2
fi
echo "--- 4) 发请求 + 抓逐位置指标 ---"
PORT=$PORT $V /home/user/ask_live.py
echo "--- 5) srv 里的 SpecDecoding ---"
grep -E 'SpecDecoding' $J/dl/srv_fp8.out | tail -4
kill $SRV 2>/dev/null
echo FP8_RUN_DONE
