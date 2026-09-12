#!/bin/bash
cd /home/user/ninfer-fusion
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
M=/home/user/models/qwen3_8_27b_nvfp4.ninfer
nohup ./build/apps/ninfer-serve "$M" --host 127.0.0.1 --port 8001 \
  --kv-dtype nvfp4 --kv-layer-storage all:nvfp4 \
  --max-context 4096 --kv-capacity 8192 --max-concurrency 8 \
  --device-state-slots 2 --host-state-slots 1 --host-kv-mib 0 \
  --log-stats-interval-ms 2000 --no-thinking --max-pending-requests 32 \
  > /tmp/serve.log 2>&1 &
echo $! > /tmp/serve.pid
sleep 30
grep -E 'listening|ready|error|terminate' /tmp/serve.log | head -3
