#!/bin/bash
# runai 流式加载（主机峰值≈单张量）+ cgroup 14GiB 上限 + 逐 2 秒记录
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
V=/home/user/vllm029/bin/python
PORT=8140
SRVLOG=$J/dl/srv_stream.out
TRACE=$J/dl/stream_trace.log
MEMDIR="/sys/fs/cgroup$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null)"
echo "=== runai 流式跑 $(date '+%H:%M:%S')  MEMDIR=$MEMDIR ==="
printf "  memory.max=%s\n" "$(cat $MEMDIR/memory.max 2>/dev/null)"
free -g | head -2

HF_HUB_OFFLINE=1 \
  setsid nohup $V -m vllm.entrypoints.openai.api_server \
    --model /home/user/models/q3nvfp4 \
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency":1}' \
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
for i in $(seq 1 220); do
  cur=$(awk '{printf "%.1f", $1/1073741824}' "$MEMDIR/memory.current" 2>/dev/null || echo "?")
  pk=$(awk '{printf "%.1f", $1/1073741824}' "$MEMDIR/memory.peak" 2>/dev/null || echo "?")
  vr=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')
  ph=$(grep -oE 'Loading runai.*|Loading .*checkpoint shards: *[0-9]+%|Using LBNHC KV cache layout|Setting attention block size|Using FlashInfer for top-p|Model loading took [0-9.]+ GiB|Capturing CUDA graph|init engine|Starting to load|Loading safetensors checkpoint' $SRVLOG 2>/dev/null | tail -1)
  echo "$(date '+%H:%M:%S') ${cur} ${pk} $((vr/1024)) ${ph}" >> $TRACE
  code=$(curl -s -m 4 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready=1; echo "  READY at $((i*2))s"; break; fi
  if ! kill -0 $SRV 2>/dev/null; then
    echo "  服务退出于 $((i*2))s"
    break
  fi
  sleep 2
done

{
  echo "=== 结果 $(date '+%H:%M:%S') ready=$ready ==="
  echo "  memory.peak = $(awk '{printf "%.2f GB", $1/1073741824}' "$MEMDIR/memory.peak" 2>/dev/null)"
  echo "  memory.events:"; cat "$MEMDIR/memory.events" 2>/dev/null | sed 's/^/    /'
  echo "  --- trace 摘要（每 20 秒一行）---"
  awk 'NR==1 || NR%10==0' $TRACE | tail -22
  echo "  --- srv 尾 6 ---"
  tail -6 $SRVLOG
} >> $TRACE

if [ "$ready" = "1" ]; then
  echo "--- 发请求 ---" >> $TRACE
  PORT=$PORT $V /home/user/ask_live.py >> $TRACE 2>&1
  grep -E 'SpecDecoding' $SRVLOG | tail -4 >> $TRACE
fi
kill $SRV 2>/dev/null
echo STREAM_RUN_DONE >> $TRACE
