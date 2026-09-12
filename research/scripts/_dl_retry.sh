#!/bin/bash
# 带重试的下载循环（镜像单大文件易断）
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
REPO=sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4
DEST=/home/user/models/q38_abl_huihui_nvfp4
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl_retry.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== 重试循环下载 起 $(date '+%m-%d %H:%M:%S') ==="
for i in $(seq 1 12); do
  echo "--- 第 $i 次尝试 $(date '+%H:%M:%S')  当前大小 $(du -sh "$DEST" 2>/dev/null | cut -f1) ---"
  python3 - "$REPO" "$DEST" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
try:
    snapshot_download(repo_id=repo, local_dir=dest, max_workers=2)
    print("SNAPSHOT_OK", flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200], flush=True)
PY
  if grep -q SNAPSHOT_OK "$LOG"; then echo "完成"; break; fi
  sleep 10
done
du -sh "$DEST" 2>/dev/null
echo "=== 结束 $(date '+%m-%d %H:%M:%S') ==="
echo RETRY_DONE
