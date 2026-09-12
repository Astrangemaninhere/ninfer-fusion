#!/bin/bash
# 探针长跑 + 统计：草稿重复率 / 前缀断点 / draft==argmax 命中分布
set -u
R=/home/user/ninfer-fusion
M=/home/user/models
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
BIN=$R/build/apps/ninfer
cd "$R/build" || exit 1
NINFER_DF2DBG=1 timeout 1200 "$BIN" "$M/qwen3_8_27b_nvfp4_dflash2.ninfer" \
  --prompt "$P" --max-new 96 --max-context 4096 --no-thinking --greedy \
  --print-token-ids --spec dflash2 --draft-tokens 7 > /home/user/probe_long.log 2>&1
echo "rc=$?"
grep -c 'df2dbg' /home/user/probe_long.log | sed 's/^/探针行数: /'
echo
python3 - <<'PY'
import re, collections
rows = []
cur = None
for line in open('/home/user/probe_long.log', encoding='utf-8', errors='replace'):
    m = re.search(r'row=(\d+) anchor=(\d+) frontier=(\d+) valid=(\d+) in_extent=(\d+) out_extent=(\d+) accepted=(\d+) count=(\d+)', line)
    if m:
        cur = dict(row=int(m.group(1)), anchor=int(m.group(2)), frontier=int(m.group(3)),
                   valid=int(m.group(4)), in_e=int(m.group(5)), out_e=int(m.group(6)),
                   accepted=int(m.group(7)), count=int(m.group(8)), cols=[])
        rows.append(cur); continue
    m = re.search(r'col=(\d+) pos=(\d+) verify=(\d+) draft=(-?\d+) argmax=(-?\d+)', line)
    if m and cur is not None:
        cur['cols'].append(dict(col=int(m.group(1)), pos=int(m.group(2)), verify=int(m.group(3)),
                                draft=int(m.group(4)), argmax=int(m.group(5))))

print("轮数 = %d" % len(rows))
if not rows: raise SystemExit
print("in_extent 分布 = %s" % dict(collections.Counter(r['in_e'] for r in rows)))
print("out_extent 分布 = %s" % dict(collections.Counter(r['out_e'] for r in rows)))
print("accepted 分布 = %s" % dict(collections.Counter(r['accepted'] for r in rows)))
print("target_valid_columns 分布 = %s" % dict(collections.Counter(r['valid'] for r in rows)))
print()
# 草稿重复率
tot_dr = 0; tot_dup = 0
first_break = collections.Counter()
match_at = collections.Counter()
ncols = collections.Counter()
for r in rows:
    cs = [c for c in r['cols'] if c['draft'] >= 0]
    if not cs: continue
    drafts = [c['draft'] for c in cs]
    ncols[len(drafts)] += 1
    tot_dr += len(drafts)
    tot_dup += sum(1 for i in range(1, len(drafts)) if drafts[i] == drafts[i-1])
    # 前缀断点：第一处 draft[c] != argmax[c+1]
    brk = None
    for i, c in enumerate(cs):
        nxt = r['cols'][i+1]['argmax'] if i+1 < len(r['cols']) else None
        if nxt is not None and c['draft'] != nxt:
            brk = i; break
    if brk is not None:
        first_break[brk] += 1
    # 命中统计：draft[c] == 目标在下一列的 argmax
    for i, c in enumerate(cs):
        if i+1 < len(r['cols']) and c['draft'] == r['cols'][i+1]['argmax']:
            match_at[i] += 1
print("草稿相邻重复: %d / %d = %.1f%%" % (tot_dup, tot_dr, 100.0*tot_dup/max(1,tot_dr)))
print("每轮草稿列数分布 = %s" % dict(ncols))
print("前缀首次断裂位置 = %s" % dict(sorted(first_break.items())))
print("draft 命中(与下一列 argmax 相等)按列 = %s" % dict(sorted(match_at.items())))
PY
echo
echo '--- 接受率/剖面 ---'
grep -oE 'dflash2 acceptance rate +[0-9.]+%' /home/user/probe_long.log | head -1
grep -oE 'accepted by pos +[0-9,]+' /home/user/probe_long.log | tail -1
grep -oE 'dflash2 rounds +[0-9]+' /home/user/probe_long.log | head -1
