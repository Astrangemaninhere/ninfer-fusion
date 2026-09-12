#!/bin/bash
# 在 cgroup 硬上限内跑 vLLM 并逐 2 秒记录：内存 + 当前阶段（加载/KV布局/后端/JIT）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8139
SRVLOG=$J/dl/srv_cap.out
TRACE=$J/dl/mem_trace.log
MEMDIR="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null)"
echo "=== cgroup 上限内跑 $(date '+%H:%M:%S')  MEMDIR=$MEMDIR ==="
cat "$MEMDIR/memory.max" 2>/dev/null | sed 's/^/  memory.max=/'
free -g | head -2

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
    > $SRVLOG 2>&1 &
SRV=$!
echo "server pid=$SRV"
echo "ts memGB peakGB vramGB phase" > $TRACE

ready=0
for i in $(seq 1 200); do
  cur=$(awk '{printf "%.1f", $1/1073741824}' "$MEMDIR/memory.current" 2>/dev/null || echo "?")
  pk=$(awk '{printf "%.1f", $1/1073741824}' "$MEMDIR/memory.peak" 2>/dev/null || echo "?")
  vr=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
  ph=$(grep -oE 'Loading (fastsafetensors )?checkpoint shards: *[0-9]+%|Using LBNHC KV cache layout|Setting attention block size|Using FlashInfer for top-p|Model loading took [0-9.]+ GiB|Capturing CUDA graph|init engine|SamplingParams' $SRVLOG 2>/dev/null | tail -1)
  echo "$(date '+%H:%M:%S') ${cur} ${pk} $((vr/1024)) ${ph}" >> $TRACE
  code=$(curl -s -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  READY at $((i*2))s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "  服务退出于 $((i*2))s —— 看 oom 事件："
    cat "$MEMDIR/memory.events" 2>/dev/null | sed 's/^/    /'
    break
  fi
  sleep 2
done

echo "=== 峰值 ==="
echo "  memory.peak = $(awk '{printf "%.2f GB", $1/1073741824}' "$MEMDIR/memory.peak" 2>/dev/null)"
cat "$MEMDIR/memory.events" 2>/dev/null | sed 's/^/  /'
echo "=== 阶段×内存（每 20 秒抽样一次）==="
awk 'NR==1 || NR%10==0' $TRACE | tail -20

if [ "$ready" = "1" ]; then
  echo "--- 发请求 ---"
  PORT=$PORT $V /home/user/ask_live.py
  grep -E 'SpecDecoding' $SRVLOG | tail -4
  kill $SRV 2>/dev/null
fi
echo CAP_RUN_DONE
