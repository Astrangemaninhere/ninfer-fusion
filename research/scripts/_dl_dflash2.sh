#!/bin/bash
# 探端点 → 用可用的那个整份下载 3.58GB 草稿 → 头校验
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
DIR=$J/data/draft_dflash2_ref
mkdir -p "$DIR"
CURL="/mnt/c/Windows/System32/curl.exe"
URL_PATH="incoai/Qwen3.8-27B-DFlash2/resolve/main/model.safetensors"
LOG=$J/dl/dl_dflash2.log
exec > >(tee -a "$LOG") 2>&1
echo "=== 探端点 $(date '+%H:%M:%S') ==="
EP=""
for e in https://hf-mirror.com https://huggingface.co; do
  code=$(timeout 40 "$CURL" -s -o /dev/null -w '%{http_code}' -I -L "$e/$URL_PATH" 2>/dev/null)
  echo "  $e -> HTTP $code"
  [ "$code" = "200" ] && { EP="$e"; break; }
done
[ -n "$EP" ] || { echo "两个端点都不可达，放弃本轮"; exit 2; }
echo "选用端点: $EP"
echo "--- 下载中（3.58GB，可续传） ---"
OUT_W="C:/Users/User/Documents/ziqinzhang/data/draft_dflash2_ref/model.safetensors"
timeout 3000 "$CURL" -L --retry 5 --retry-delay 3 -C - --no-progress-meter \
  -o "$OUT_W" "$EP/$URL_PATH"
echo "curl rc=$?"
ls -l --time-style=+%H:%M "$DIR/model.safetensors" 2>/dev/null | tail -1
echo "--- 头校验 ---"
python3 - "$DIR/model.safetensors" <<'PY'
import sys, json, struct, pathlib
p = pathlib.Path(sys.argv[1]); sz = p.stat().st_size
with open(p,'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
end = 8 + n + max((v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__'), default=0)
print("size=%d declared_end=%d => %s" % (sz, end, "COMPLETE" if end<=sz else "TRUNCATED(%.1f%%)" % (100.0*sz/end)))
PY
echo DL_DFLASH2_DONE
