#!/usr/bin/env bash
# serve_muse.sh - ninfer 模型一键启动+轮询 (KV 实验参数透传)
# 用法: ./serve_muse.sh [muse|qwen27|qwen27-dflash2] [port] [kv-layer-storage]
set -u
MODEL=${1:-muse}
PORT=${2:-8001}
KVARG=${3:-}
case "$MODEL" in
  muse)      ART=/home/user/models/muse_glimmer_30b_nvfp4.ninfer;  TAG=muse_glimmer_30b ;;
  qwen27)    ART=/home/user/models/qwen3_8_27b_nvfp4.ninfer;       TAG=qwen3_8_27b_nvfp4 ;;
  qwen27-dflash2) ART=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer; TAG=qwen3_8_27b_nvfp4_dflash2 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
LOG=/home/user/serve_${TAG}${KVARG:+_$KVARG}.log
BIN=/home/user/ninfer-fusion/build/apps/ninfer-serve
[ -f "$ART" ] || { echo "artifact missing: $ART"; exit 2; }
[ -f "$BIN" ] || { echo "binary missing"; exit 2; }

for pid in $(pgrep -f "ninfer-serve.*$TAG"); do
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
  case "$cmd" in *$TAG*) kill "$pid" 2>/dev/null && echo "killed stale $pid";; esac
done
sleep 3

KV_OPTS="--kv-dtype bf16"
[ -n "$KVARG" ] && KV_OPTS="--kv-layer-storage $KVARG"
rm -f "$LOG"
nohup env LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64 "$BIN" "$ART" --port "$PORT" \
  $KV_OPTS --max-context 2048 --kv-capacity 2048 --greedy --max-concurrency 1 \
  > "$LOG" 2>&1 &
echo "serve($TAG $KVARG) pid $! -> $LOG"

for i in $(seq 1 40); do
  sleep 15
  if grep -qiE 'listening|http://' "$LOG"; then
    echo "[ok] listening after ~$((i*15))s"
    grep -E 'load|error' "$LOG" | tail -3
    exit 0
  fi
  if grep -qiE 'error|aborted' "$LOG"; then
    echo "[fail] at ~$((i*15))s"
    grep -E 'error' "$LOG" | tail -3
    exit 1
  fi
done
echo "[timeout]"
tail -3 "$LOG"
exit 3
