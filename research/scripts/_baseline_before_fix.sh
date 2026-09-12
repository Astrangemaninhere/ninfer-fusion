#!/bin/bash
# 修复前基准表（canonical "before"）：
#   A) zh prompt：plain / dflash2 / mtp3 的 token 流对比 + 接受率 + 解码 tok/s
#   B) 非 128 对齐长 prompt：chunk 128 vs 4096（灵敏探针，当前应 DIFFER at 0）
#   C) 128 对齐长 prompt：chunk 128 vs 4096（应 IDENTICAL，作对照）
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG=$J/dl/baseline_before_fix.log
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== baseline BEFORE fix $(date '+%F %H:%M:%S') ==="
python3 - <<'PY'
import pathlib
SEG = "杭州地处长江三角洲南翼，浙江省北部，北靠天目山，南临钱塘江，水网密布、湖荡众多。"
pathlib.Path("/tmp/p_long_unaligned.txt").write_text(SEG * 60, encoding="utf-8")   # 1800 tok, 非 128 倍数
pathlib.Path("/tmp/p_long_aligned.txt").write_text(SEG * 64, encoding="utf-8")     # 1920 = 15*128
print("prompts written")
PY
cd $R/build || exit 3
A=/home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer
ZH='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
LU="$(cat /tmp/p_long_unaligned.txt)"
LA="$(cat /tmp/p_long_aligned.txt)"

run() {  # tag maxnew prompt extra...
  local tag=$1 mn=$2 p=$3; shift 3
  local log=/home/user/bl_$tag.log
  timeout 900 ./apps/ninfer "$A" --prompt "$p" --max-new "$mn" --max-context 8192 \
      --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1
  grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
      sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//' > /tmp/bl_$tag.ids
  local pos; pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local rate; rate=$(grep -oE 'acceptance rate[[:space:]]+[0-9.]+%' "$log" | tail -1 | sed 's/.* //')
  local dps; dps=$(grep -oE 'decode speed[[:space:]]+[0-9.]+' "$log" | tail -1 | sed 's/.* //')
  echo "  [$tag] n=$(wc -w < /tmp/bl_$tag.ids) rate=$rate pos=[$pos] decode=${dps}tok/s"
}

echo "--- A) zh ---"
run zh_plain  96 "$ZH"
run zh_df2    96 "$ZH" --spec dflash2
run zh_mtp3   96 "$ZH" --spec mtp --draft-tokens 3
echo "--- B) 非对齐长 prompt（灵敏） ---"
run lu_c128  32 "$LU" --prefill-chunk 128
run lu_c4096 32 "$LU" --prefill-chunk 4096
echo "--- C) 128 对齐长 prompt（对照） ---"
run la_c128  32 "$LA" --prefill-chunk 128
run la_c4096 32 "$LA" --prefill-chunk 4096

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
def ids(t):
    p = pathlib.Path(f"/tmp/bl_{t}.ids")
    return p.read_text().split() if p.exists() else []
def cmp(a, b):
    A, B = ids(a), ids(b)
    if not A or not B:
        return f"{a} vs {b}: NO DATA"
    n = min(len(A), len(B))
    i = next((x for x in range(n) if A[x] != B[x]), n)
    if i == n and len(A) == len(B):
        return f"{a} vs {b}: IDENTICAL ({len(A)} tok)"
    return (f"{a} vs {b}: DIFFER at {i}/{n}  A={A[max(0,i-2):i+2]} B={B[max(0,i-2):i+2]}")
print()
print("== 基准表（修复前）==")
for a, b in (("zh_plain", "zh_df2"), ("zh_plain", "zh_mtp3"),
             ("lu_c128", "lu_c4096"), ("la_c128", "la_c4096")):
    print("  " + cmp(a, b))
PY
echo BASELINE_BEFORE_DONE
