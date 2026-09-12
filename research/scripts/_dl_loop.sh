#!/bin/bash
# 持久下载循环：一直重试直到 snapshot 成功（供长时间挂着）
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
REPO=sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4
DEST=/home/user/models/q38_abl_huihui_nvfp4
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl_loop.log
echo "=== 持久下载循环 起 $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
i=0
while [ "$i" -lt 400 ]; do
  i=$((i+1))
  echo "--- 第 $i 次 $(date '+%H:%M:%S') 当前 $(du -sh "$DEST" 2>/dev/null | cut -f1) ---" >> "$LOG"
  out=$(python3 - "$REPO" "$DEST" 2>&1 <<'PY'
import sys, time
from huggingface_hub import snapshot_download
try:
    snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2], max_workers=2)
    print("SNAPSHOT_OK")
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:160])
PY
)
  echo "$out" >> "$LOG"
  case "$out" in *SNAPSHOT_OK*) echo "=== 完成 $(date '+%H:%M:%S') ===" >> "$LOG"; break;; esac
  sleep 5
done
echo "LOOP_DONE" >> "$LOG"
