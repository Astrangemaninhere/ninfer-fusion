#!/bin/bash
# Acceptance-rate verdict for the W9 dflash2 draft (per _TODO.md 112).
#
# Two boundary lessons are baked in here (both found by a smoke test today, not
# in theory):
#   * `eval_ddtree.py` is a WINDOWS python script — it resolves a hardcoded
#     Windows TARGET_DIR for the teacher's lm_head/embed and uses the Windows
#     torch, so WSL's python cannot run it;
#   * WSL environment variables do NOT cross into Windows processes, so
#     DF2_CACHE must be set on the Windows side — that is why the runs live in
#     `_df2_accept_eval.bat` and this script only orchestrates + judges.
#
#   GATE (pre-registered): teacher hit@1 >= 0.35 AND (hit@4 - hit@1) >= 0.12
#   => build the beam-L DDTree; otherwise the tree is not worth its nodes.
#
# Usage (GPU window; training must be paused): bash _df2_accept_eval.sh [ckpt] [anchors]
set -u
J=/mnt/c/Users/User/Documents/ziqinzhang
LOG="$J/data/df2_accept_eval.log"
OUT="$J/_collab/M_accept_eval.md"
CKPT=${1:-data\dflash2_ckpts\step_006000.pt}
ANCHORS=${2:-1200}
WIN_CKPT="${CKPT//\//\\}"

cd "$J" || exit 2
[ -f "${WIN_CKPT//\\//}" ] || { echo "FATAL: checkpoint missing: $CKPT"; exit 2; }
if pgrep -af 'train_dflash2' | grep -v $$ > /dev/null; then
  echo "ABORT: training is alive — the GPU must be free (kill it deliberately first)"; exit 3
fi
: > "$LOG"

cmd.exe /c "$(wslpath -w "$J")\\_df2_accept_eval.bat" "$WIN_CKPT" "$ANCHORS"
rc=$?
echo "bat exit=$rc; log=$LOG"

python3 - "$LOG" "$OUT" "$CKPT" "$ANCHORS" <<'PY'
import re, sys
logp, out, ckpt, anchors = sys.argv[1:5]
lines_in = open(logp, encoding='utf-8', errors='replace').read().splitlines()
blocks, cur, tails = {}, None, {}
for ln in lines_in:
    m = re.match(r'^=== (chain|beam|tree)', ln)
    if m:
        cur = m.group(1); blocks[cur] = []; tails[cur] = []
    elif cur is not None:
        blocks[cur].append(ln)
        tails[cur] = (tails[cur] + [ln])[-25:]

def decision(lines):
    for ln in reversed(lines):
        m = re.search(r'DECISION (\S+) row-wise\s*: hit@1=([\d.]+) hit@4-hit@1=([+-]?[\d.]+)', ln)
        if m:
            return m.group(1), float(m.group(2)), float(m.group(3))
    return None, None, None

res = {k: decision(v) for k, v in blocks.items()}
res.get('beam', (None, None, None))
out_lines = ['# 接受率判定 (预注册规则, _TODO.md 112)', '', 'ckpt=%s anchors=%s' % (ckpt, anchors)]
for k in ('chain', 'beam', 'tree'):
    src, h1, d = res.get(k, (None, None, None))
    out_lines.append('- %-6s source=%s hit@1=%s hit@4-hit@1=%s' % (k, src, h1, d))
_, h1, d = res.get('chain', (None, None, None))
if h1 is None:
    out_lines.append('**VERDICT: EVAL_FAILED** (无 DECISION 行 — 缺失数据不得当结论)')
    # include each phase's last lines so the failure is diagnosable without the log
    for k, t in tails.items():
        out_lines += ['', '### %s 尾部' % k] + ['    ' + x[:160] for x in t[-8:]]
else:
    gate = (h1 >= 0.35) and (d is not None and d >= 0.12)
    out_lines.append('- GATE (hit@1>=0.35 且 headroom>=0.12): %s' % ('PASS' if gate else 'FAIL'))
    out_lines.append('**VERDICT: %s**' % ('BUILD_TREE' if gate else 'TREE_NOT_WORTH_IT'))
    tb, tt = res.get('beam', (None, None, None)), res.get('tree', (None, None, None))
    if tb[1] is not None and tt[1] is not None:
        out_lines.append('- beam 相对 chain: hit@1 %+.4f, headroom %+.4f' % (tb[1] - h1, tb[2] - d))
        out_lines.append('- tree 相对 beam : hit@1 %+.4f, headroom %+.4f' % (tt[1] - tb[1], tt[2] - tb[2]))
    out_lines.append('- 口径: `--teacher` (语料口径会低估计 ~18%); 仍是离线近似; 真机接受率需在引擎里用同一 ckpt 复测。')
open(out, 'w', encoding='utf-8').write('\n'.join(out_lines) + '\n')
print('\n'.join(out_lines))
PY
echo "written: $OUT"
echo DF2_ACCEPT_EVAL_DONE
