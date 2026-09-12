#!/bin/bash
# 下载破禁版 bf16 源（通用 convert.py 的 --model）
set -u
export HF_ENDPOINT=https://hf-mirror.com
REPO=huihui-ai/Huihui-Qwen3.8-27B-abliterated
DEST=/home/user/models/q38_abl_huihui_bf16
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl_bf16.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== 下载 $REPO -> $DEST 起 $(date '+%m-%d %H:%M:%S') ==="
mkdir -p "$DEST"
python3 - "$REPO" "$DEST" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
try:
    print("OK ->", snapshot_download(repo_id=repo, local_dir=dest, max_workers=6), flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, e, flush=True)
print(f"耗时 {time.time()-t0:.0f}s", flush=True)
PY
echo "--- 落盘 ---"
du -sh "$DEST" 2>/dev/null
ls "$DEST" | head -25
echo "=== 结束 $(date '+%m-%d %H:%M:%S') ==="
echo BF16_DL_DONE
