#!/bin/bash
# E1 判别（A1 提出）：**不投机**的前提下，只改 `--prefill-chunk` ⇒ 若 token id 不同，
# 就坐实"贪心输出随 T/batching 变化"，即等价性缺口来自 kernel 实例化而非草稿。
# 三种 chunk 同一 prompt/温度，比生成的 token id 序列。
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
R=/home/user/ninfer-fusion
CLI=$R/build/apps/ninfer
MODEL=${MODEL:-/home/user/models/qwen3_8_27b_nvfp4.ninfer}
LOG=$J/dl/chunk_equiv.log
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'

if pgrep -f 'train_dflash2' >/dev/null 2>&1; then echo "training running - need GPU alone" | tee -a "$LOG"; exit 3; fi

one() {  # label chunk
  local label=$1 chunk=$2
  local log=/home/user/ce_$label.log
  : > "$log"
  ( cd $R/build && timeout 900 "$CLI" "$MODEL" --prompt "$P" --max-new 48 --max-context 2048 \
      --no-thinking --greedy --print-token-ids --prefill-chunk "$chunk" > "$log" 2>&1 )
  local rc=$?
  local ids; ids=$(grep -E '^tokens[[:space:]]+generated ids' "$log" | tail -1 | \
                   sed -E 's/^tokens[[:space:]]+generated ids[[:space:]]*//')
  echo "  [$label chunk=$chunk] rc=$rc n=$(echo $ids | wc -w)"
  echo "$label|$chunk|$rc|$ids" >> /tmp/ce_rows
}

: > /tmp/ce_rows
echo "=== chunk-equivalence (T 依赖) $(date '+%F %H:%M:%S') ===" | tee -a "$LOG"
one c128   128
one c512   512
one c2048  2048

python3 - <<'PY' | tee -a "$LOG"
import pathlib, time
rows = {}
for line in pathlib.Path("/tmp/ce_rows").read_text().splitlines():
    lab, chunk, rc, ids = (line.split("|", 3) + ["", "", ""])[:4]
    rows[lab] = (chunk, ids.split())
def cmp(a, b):
    ca, A = rows.get(a, ("", [])); cb, B = rows.get(b, ("", []))
    if not A or not B: return "%s vs %s: NO DATA" % (a, b)
    n = min(len(A), len(B)); i = next((k for k in range(n) if A[k] != B[k]), n)
    if i == n and len(A) == len(B):
        return "chunk %s vs %s: IDENTICAL (%d tokens)" % (ca, cb, len(A))
    return ("chunk %s vs %s: **DIFFER at token %d** (A[%d]=%s B[%d]=%s; lens %d/%d)"
            % (ca, cb, i, i, A[i], i, B[i], len(A), len(B)))
out = ["", "## E1 判别：plain 输出是否随 --prefill-chunk 变化 (%s)" % time.strftime("%F %H:%M"), "",
       "- " + cmp("c128", "c2048"),
       "- " + cmp("c512", "c2048"),
       "",
       "判读：**plain + 贪心**下输出若随 chunk 变化 ⇒ 引擎的贪心结果依赖 batching/分块",
       "（即 kernel 实例化带来的数值差异足以翻转 argmax）⇒ 投机 verify（T=width≠1）与 plain（T=1）本就不可能逐 token 一致，",
       "接受率的上限被**实现数值差**封住，而不是草稿质量。若完全相同 ⇒ 排除该假设，转查 dflash2 的 rope_delta 与 KV 量化路径。"]
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_chunk_equivalence.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
PY
echo CHUNK_EQUIV_DONE | tee -a "$LOG"
