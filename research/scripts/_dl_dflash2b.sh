#!/bin/bash
# 改用 hf-mirror.com（接受 3xx），整份下载 3.58GB 草稿 + 头校验
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
DIR=$J/data/draft_dflash2_ref
CURL="/mnt/c/Windows/System32/curl.exe"
REL="incoai/Qwen3.8-27B-DFlash2/resolve/main/model.safetensors"
EP=https://hf-mirror.com
LOG=$J/dl/dl_dflash2b.log
exec > >(tee -a "$LOG") 2>&1
echo "=== hf-mirror 下载 $(date '+%H:%M:%S') ==="
code=$(timeout 40 "$CURL" -s -o /dev/null -w '%{http_code}' -I -L "$EP/$REL" 2>/dev/null)
echo "  端点检查 HTTP $code"
case "$code" in 2*|3*) echo "  可用" ;; *) echo "  不可用，退出"; exit 2;; esac
OUT="C:/Users/User/Documents/ziqinzhang/data/draft_dflash2_ref/model.safetensors"
echo "--- 下载中 ---"
timeout 3300 "$CURL" -L --retry 8 --retry-delay 3 --retry-all-errors -C - --no-progress-meter \
  -o "$OUT" "$EP/$REL"
echo "curl rc=$?"
ls -l --time-style=+%H:%M "$DIR/model.safetensors" 2>/dev/null | tail -1
python3 - "$DIR/model.safetensors" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
print("size=%d declared_end=%d => %s" % (sz, end, "COMPLETE" if end<=sz else "TRUNCATED(%.1f%%)" % (100.0*sz/end)))
PY
echo DL2B_DONE
