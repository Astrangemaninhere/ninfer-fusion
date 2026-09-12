#!/bin/bash
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 备份编译前的二进制（作为修复前参照） ==="
cp -f "$R/build/apps/ninfer" /home/user/ninfer_pre_fix 2>/dev/null && echo "  已备份 ninfer -> /home/user/ninfer_pre_fix"
cp -f "$R/build/apps/ninfer-serve" /home/user/ninfer-serve_pre_fix 2>/dev/null && echo "  已备份 ninfer-serve -> /home/user/ninfer-serve_pre_fix"
ls -l --time-style=+%m-%d_%H:%M /home/user/ninfer_pre_fix /home/user/ninfer-serve_pre_fix 2>/dev/null | awk '{print "  ", $6, $5, $NF}'
echo "  （对照基准：18:08 的二进制 = 含补丁A/SWA/S28/S51/Muse-policy，但**不含** dspark 列偏移与 MTP 掩码修复）"
echo
echo "=== 写综合验证脚本（编译完成后跑） ==="
cat > "$J/_post_fix_verify.sh" <<'EOS'
#!/bin/bash
# 修复前(旧二进制) vs 修复后(新二进制) 的双重对照：
#   (1) dspark 位置剖面与接受率（应显著上升：修复前 accept=11.22% / p0=15/56=26.8%）
#   (2) mtp3 的 token id 与接受率（MTP 修复在 next>steps 时应逐位等价 ⇒ token id 应相同）
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
NEW=$R/build/apps/ninfer
OLD=/home/user/ninfer_pre_fix
LOG=$J/dl/post_fix_verify.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
export PATH="/home/user/.local/bin:$PATH"
exec >> "$LOG" 2>&1
echo "=================================================================="
echo "=== post-fix verify $(date '+%F %H:%M:%S') ==="

wait_idle() {
  local t0=$(date +%s)
  while pgrep -f 'bin/nvcc|cc1plus' >/dev/null 2>&1; do
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && { echo "TIMEOUT 等编译"; return 1; }
    sleep 20
  done
  return 0
}
wait_idle || exit 2
ls -l --time-style=+%H:%M "$NEW" | awk '{print "  新二进制:", $6, $5}'

one() {  # tag bin extra...
  local tag=$1 bin=$2; shift 2
  local log=/home/user/pfv_$tag.log
  ( cd $R/build && timeout 900 "$bin" /home/user/models/${MODEL_NAME} --prompt "$P" \
      --max-new 96 --max-context 4096 --no-thinking --greedy --print-token-ids "$@" > "$log" 2>&1 )
  local ids; ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
                   sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  local rate; rate=$(grep -oE 'spec_accept_rate=[0-9.]+' "$log" | tail -1)
  local pos;  pos=$(grep -oE 'accepted by pos[[:space:]]+[0-9,]+' "$log" | tail -1 | sed 's/.*pos *//')
  local tpr;  tpr=$(grep -oE '[0-9.]+tok/round' "$log" | tail -1)
  echo "  [$tag] n=$(echo $ids | wc -w) $rate $tpr pos=[$pos]"
  echo "$tag|$ids|$rate|$pos" >> /tmp/pfv_rows
}

: > /tmp/pfv_rows
echo
echo "--- (1) dspark：修复前 vs 修复后 ---"
MODEL_NAME=qwen3_8_27b_nvfp4_dspark.ninfer
[ -x "$OLD" ] && one dspark_OLD "$OLD" --spec dflash --draft-tokens 7
one dspark_NEW "$NEW" --spec dflash --draft-tokens 7
echo
echo "--- (2) mtp3：等价性对照（token id 应相同） ---"
MODEL_NAME=qwen3_8_27b_nvfp4.ninfer
[ -x "$OLD" ] && one mtp_OLD "$OLD" --spec mtp --draft-tokens 3
one mtp_NEW "$NEW" --spec mtp --draft-tokens 3
echo
echo "--- (3) dflash2：对照（列本就对，应基本不变） ---"
MODEL_NAME=qwen3_8_27b_nvfp4_dflash2.ninfer
[ -x "$OLD" ] && one df2_OLD "$OLD" --spec auto
one df2_NEW "$NEW" --spec auto

python3 - <<'PY' 2>&1 | tee -a "$LOG"
import pathlib
rows = {}
for line in pathlib.Path("/tmp/pfv_rows").read_text().splitlines():
    tag, ids, rate, pos = (line.split("|", 3) + ["", "", ""])[:4]
    rows[tag] = (ids.split(), rate, pos)
def cmp(a, b):
    A, ar, ap = rows.get(a, ([], "", "")); B, br, bp = rows.get(b, ([], "", ""))
    if not A or not B: return "%s vs %s: NO DATA" % (a, b)
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    same = (i == n and len(A) == len(B))
    return ("%s vs %s: %s | accept %s -> %s | p0 %s -> %s"
            % (a, b, "IDENTICAL" if same else "DIFFER at %d" % i, ar or "-", br or "-",
               (ap.split(",")[0] if ap else "-"), (bp.split(",")[0] if bp else "-")))
out = ["", "## 修复前后对照 (%s)" % __import__("time").strftime("%F %H:%M"), "",
       "- " + cmp("dspark_OLD", "dspark_NEW") + "   ← 判据：accept/p0 应显著上升",
       "- " + cmp("mtp_OLD", "mtp_NEW") + "   ← 判据：应 **IDENTICAL**（零风险等价改动）",
       "- " + cmp("df2_OLD", "df2_NEW") + "   ← 判据：应相同（dflash2 的列本来就对）"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_post_fix_verify.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo POST_FIX_VERIFY_DONE | tee -a "$LOG"
EOS
bash -n "$J/_post_fix_verify.sh" && echo "  脚本 SYNTAX_OK"
