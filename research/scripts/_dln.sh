#!/bin/bash
# 关闭 Xet（CAS）走经典 HTTP/LFS 重试下载；Xet 端点 401 是本仓大分片拿不到的原因
set -u
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
REPO=sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4
DEST=/home/user/models/q38_abl_huihui_nvfp4
LOG=/mnt/c/Users/User/Documents/ziqinzhang/dl/hf_dl_noxet.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== no-xet 下载 $REPO 起 $(date '+%m-%d %H:%M:%S') ==="
python3 - "$REPO" "$DEST" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
try:
    print("OK ->", snapshot_download(repo_id=repo, local_dir=dest, max_workers=4), flush=True)
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:300], flush=True)
print(f"耗时 {time.time()-t0:.0f}s", flush=True)
PY
du -sh "$DEST" 2>/dev/null
ls -l "$DEST" | awk '{printf "%12s  %s\n", $5, $9}' | tail -8
echo "=== 结束 $(date '+%m-%d %H:%M:%S') ==="
echo NOXET_DONE
