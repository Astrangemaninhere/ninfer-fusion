#!/bin/bash
# 走镜像下载破禁版 NVFP4（lm_head 与 mtp 均留 bf16）
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=0
REPO=sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4
DEST=/home/user/models/q38_abl_huihui_nvfp4
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== 下载 $REPO 起 $(date '+%m-%d %H:%M:%S') ==="
mkdir -p "$DEST"
python3 - "$REPO" "$DEST" <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
try:
    p = snapshot_download(repo_id=repo, local_dir=dest,
                          max_workers=8, resume_download=True)
    print("OK ->", p)
except Exception as e:
    print("FAIL:", type(e).__name__, e)
print(f"耗时 {time.time()-t0:.0f}s")
PY
echo "--- 落盘结果 ---"
du -sh "$DEST" 2>/dev/null
ls "$DEST" | head -25
echo "=== 下载阶段结束 $(date '+%m-%d %H:%M:%S') ==="
echo HF_DL_DONE
