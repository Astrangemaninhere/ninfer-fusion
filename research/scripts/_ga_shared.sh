#!/bin/bash
# 共享路径判别：同一 prompt 下 plain / dflash2(K=3) / mtp(K=3) 三条流比较。
#   mtp 与 dflash2 在同一 token 位置首领偏离 => 漂移在共享的 verify/状态机（与草稿后端无关）
#   两者偏离位置不同                => 与各后端的草稿/状态处理有关
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/ga_shared.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== shared-path drift probe $(date '+%F %H:%M:%S') ==="
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

one() {
  local tag=$1; shift
  local log=/home/user/sh_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/sh_$tag.ids
  echo "  [$tag] n=$(wc -w < /tmp/sh_$tag.ids)"
}
one plain
one df2k3 --spec dflash2 --draft-tokens 3
one mtpk3 --spec mtp --draft-tokens 3

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/sh_{t}.ids")
    return p.read_text().split() if p.exists() else []
ref = ids("plain")
def first_diff(other):
    B = ids(other)
    if not B:
        return None, None, None
    n = min(len(ref), len(B))
    i = next((x for x in range(n) if ref[x] != B[x]), n)
    return (i, "IDENTICAL" if (i == n and len(ref) == len(B)) else "DIFFER", B)
print()
print(f"plain n={len(ref)}")
for tag in ("df2k3", "mtpk3"):
    i, verdict, B = first_diff(tag)
    if verdict is None:
        print(f"  {tag}: NO DATA"); continue
    print(f"  {tag}: {verdict}" + ("" if verdict == "IDENTICAL"
          else f" at {i}/{min(len(ref), len(B))}"))
    if verdict != "IDENTICAL":
        print(f"      plain[{max(0,i-3)}:{i+3}] = {ref[max(0,i-3):i+3]}")
        print(f"      {tag}[{max(0,i-3)}:{i+3}] = {B[max(0,i-3):i+3]}")
PY
echo SHARED_PROBE_DONE
