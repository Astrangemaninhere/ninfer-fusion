#!/usr/bin/env python3
"""判定草稿是否"整体偏移 k 位"：对每个 k，统计 draft[c] == argmax[c+1+k] 的命中数。
若某个 k>0 的命中远高于 k=0，则草稿头的时间口径偏移了 k 位（= T3 的 legacy 假说）。"""
import re, collections

rows, cur = [], None
for line in open('/home/user/probe_long.log', encoding='utf-8', errors='replace'):
    m = re.search(r'row=(\d+) anchor=(\d+) frontier=(\d+) valid=(\d+) in_extent=(\d+) out_extent=(\d+) accepted=(\d+) count=(\d+)', line)
    if m:
        cur = dict(in_e=int(m.group(5)), cols=[]); rows.append(cur); continue
    m = re.search(r'col=(\d+) pos=(\d+) verify=(\d+) draft=(-?\d+) argmax=(-?\d+)', line)
    if m and cur is not None:
        cur['cols'].append(dict(col=int(m.group(1)), pos=int(m.group(2)),
                                draft=int(m.group(4)), argmax=int(m.group(5))))

# 只取真的有草稿的轮
live = [r for r in rows if r['in_e'] > 0]
print("有草稿的轮数 = %d（总轮数 %d）" % (len(live), len(rows)))

print("\n=== 命中矩阵：draft[c] == argmax[c+1+k]，k 为偏移 ===")
print("  k :  命中数 / 可比较数   (命中率)")
for k in (-1, 0, 1, 2, 3):
    hit = tot = 0
    for r in live:
        cs = [c for c in r['cols'] if c['draft'] >= 0]
        for i, c in enumerate(cs):
            j = i + 1 + k
            if 0 <= j < len(r['cols']):
                tot += 1
                if c['draft'] == r['cols'][j]['argmax']:
                    hit += 1
    print("  %+d :  %4d / %4d          %5.1f%%" % (k, hit, tot, 100.0*hit/max(1,tot)))

print("\n=== 另一口径：draft[c] 是否等于『该位置的目标 argmax』的另一种取法 ===")
print("（用 draft[c] 所在列自己的 argmax 作为参照，即 draft[c]==argmax[c]）")
hit = tot = 0
for r in live:
    cs = [c for c in r['cols'] if c['draft'] >= 0]
    for i, c in enumerate(cs):
        j = i + 1
        if j < len(r['cols']):
            tot += 1
            if c['draft'] == r['cols'][j]['argmax']:
                hit += 1
print("  draft[c] == argmax[c+1] : %d / %d = %.1f%%" % (hit, tot, 100.0*hit/max(1,tot)))

print("\n=== verify 与 draft 的对应（确认无 extent 截断）===")
ok = bad = 0
for r in live:
    cs = [c for c in r['cols']]
    for i in range(len(cs)-1):
        if cs[i]['draft'] >= 0:
            if r['cols'][i+1]['verify'] == cs[i]['draft']:
                ok += 1
            else:
                bad += 1
print("  verify[c+1]==draft[c] 成立 %d 次，不成立 %d 次" % (ok, bad))

print("\n=== 首颗草稿(draft[0])错位检查：它是否等于目标在 pos0/pos1/pos2 的 argmax ===")
cnt = collections.Counter()
tot = 0
for r in live:
    cs = [c for c in r['cols'] if c['draft'] >= 0]
    if not cs: continue
    d0 = cs[0]['draft']; tot += 1
    for k in (-1, 0, 1, 2):
        j = 1 + k
        if 0 <= j < len(r['cols']) and d0 == r['cols'][j]['argmax']:
            cnt[k] += 1
print("  总数 %d，draft[0] 命中偏移 k 的次数: %s" % (tot, dict(sorted(cnt.items()))))
