#!/bin/bash
# 落地 GDN conv column 修复后：重编 + 终检。
# 终检判据：
#   1) spec 流 vs plain **逐位一致**（偏移消除）
#   2) 接受率剖面是否变化
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/gdn_fix_verify.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== gdn conv fix: build + verify $(date '+%F %H:%M:%S') ==="
cd "$R" || exit 3
echo "--- touch gdn_conv.cuh 的 includer ---"
for f in src/ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu \
         src/ops/gdn_input_proj/w8/w8_gdn_input_decode.cu \
         src/ops/gdn_input_proj/gdn_projected_conv.cu \
         src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu \
         src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu; do
  [ -f "$f" ] && { touch "$f"; echo "  touched $f"; }
done

cd "$R/build" || exit 3
make ninfer -j2 2>&1 | tail -8
rc=${PIPESTATUS[0]}
echo "make rc=$rc"
[ "$rc" -ne 0 ] && { echo BUILD_FAIL; exit 1; }
ls -l --time-style=+%H:%M ./apps/ninfer

A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
one() {  # tag extra...
  local tag=$1; shift
  local log=/home/user/gf_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$P" --max-new 96 --max-context 4096 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/gf_$tag.ids
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local rate; rate=$(grep -oE 'acceptance rate[[:space:]]+[0-9.]+%' "$log" | tail -1 | sed 's/.* //')
  local al; al=$(grep -oE 'acceptance length[[:space:]]+[0-9.]+' "$log" | tail -1 | sed 's/.* //')
  echo "  [$tag] n=$(wc -w < /tmp/gf_$tag.ids) rate=$rate AL=$al pos=[$pos]"
}
one plain
one dflash2 --spec dflash2
one mtp3    --spec mtp --draft-tokens 3

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/gf_{t}.ids")
    return p.read_text().split() if p.exists() else []
ref = ids("plain")
print()
print(f"plain n={len(ref)}")
for t in ("dflash2", "mtp3"):
    B = ids(t)
    if not B:
        print(f"  {t:>8}: NO DATA"); continue
    n = min(len(ref), len(B))
    i = next((x for x in range(n) if ref[x] != B[x]), n)
    if i == n and len(ref) == len(B):
        print(f"  {t:>8}: **IDENTICAL**  <-- 偏移消除（G-A 通过）")
    else:
        print(f"  {t:>8}: DIFFER at {i}/{n}  plain={ref[max(0,i-2):i+2]} spec={B[max(0,i-2):i+2]}")
PY
echo GDN_FIX_VERIFY_DONE
