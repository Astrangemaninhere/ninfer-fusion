#!/bin/bash
# 核验 bf16head artifact 结构，并跑 dflash2/MTP 接受率对照
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
NEW=/home/user/models/qwen3_8_27b_nvfp4_dflash2_bf16head.ninfer
OLD=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
LOG=$J/dl/bf16head_ab.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== bf16 头判决 起 $(date '+%m-%d %H:%M:%S') ==="
ls -l "$NEW" "$OLD" 2>&1 | awk '{print "  "$5"  "$9}'

echo
echo "--- 结构核验（新 artifact 头对象格式 / payload_start / 范围）---"
python3 - "$NEW" <<'PY'
import json, struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    f.read(8)
    (n,) = struct.unpack("<Q", f.read(8))
    d = json.loads(f.read(n).decode())
    f.seek(0, 2)
    total = f.tell()
M = (16 + n + 4095) // 4096 * 4096
objs = d["objects"]
end = max(o["offset"] + o["bytes"] for o in objs)
o = [x for x in objs if x["name"] == "text/output_head"][0]
print(f"  json_len={n} payload_start={M} 对象数={len(objs)} payload_end={end} 文件={total}")
print(f"  text/output_head -> format={o['format']} layout={o['layout']} bytes={o['bytes']}")
print(f"  范围自洽: {'OK' if M + end <= total else 'FAIL'}")
print(f"  identity={d['identity']}")
PY

cd "$R/build" || exit 3
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
printf "  %-22s %-9s %-22s %-11s %s\n" 臂 接受率 位置剖面 解码 轮数
run() {
  local label="$1"; local art="$2"; shift 2
  timeout 900 ./apps/ninfer "$art" --prompt "$P" --max-new 96 --max-context 4096 \
    --no-thinking --greedy "$@" > $J/dl/bh_$label.log 2>&1
  local a p sp rd
  a=$(grep -m 1 -E 'dflash2 acceptance rate|mtp acceptance rate' $J/dl/bh_$label.log | grep -oE '[0-9.]+%')
  p=$(grep -m 1 -E 'dflash2 accepted by pos|mtp accepted by pos' $J/dl/bh_$label.log | sed 's/.*pos *//')
  sp=$(grep -m 1 'decode speed' $J/dl/bh_$label.log | grep -oE '[0-9.]+ tok/s')
  rd=$(grep -m 1 -E 'dflash2 rounds|mtp rounds' $J/dl/bh_$label.log | grep -oE '[0-9]+')
  if [ -z "${a:-}" ]; then a="ERR: $(tail -c 200 $J/dl/bh_$label.log | tr '\n' ' ' | tail -c 120)"; fi
  printf "  %-22s %-9s %-22s %-11s %s\n" "$label" "$a" "${p:-?}" "${sp:-?}" "${rd:-?}"
}
run old_dflash2 "$OLD" --spec dflash2
run new_bf16head_dflash2 "$NEW" --spec dflash2
run old_mtp     "$OLD" --spec mtp --draft-tokens 3 --lm-head-draft
run new_bf16head_mtp "$NEW" --spec mtp --draft-tokens 3 --lm-head-draft
run new_bf16head_plain "$NEW"
echo "=== 完成 $(date '+%m-%d %H:%M:%S') ==="
echo BF16HEAD_AB_DONE
