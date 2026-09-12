#!/bin/bash
# 草稿头边项是否在起作用：NINFER_DF2SEL 探针
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
export PATH=/home/user/.local/bin:$PATH
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 3
NINFER_DF2SEL=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 64 --max-context 4096 --no-thinking --greedy \
  --spec dflash2 --draft-tokens 7 > "$J/dl/sel_probe.log" 2>&1
echo "rc=$? 行数=$(grep -c df2sel $J/dl/sel_probe.log)"

python3 - <<'PY'
import pathlib, re, statistics, collections
p = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/sel_probe.log")
RX = re.compile(r"\[df2sel\] s=(\d+) pred=(-?\d+) chosen=(-?\d+) tok=(-?\d+) E=(-?[\d.]+) u=(-?[\d.]+) "
                r"pair=(-?[\d.]+) uspan=(-?[\d.]+) pairspan=(-?[\d.]+) Earg=(-?\d+) Uarg=(-?\d+)")
rows = []
for line in p.read_text(errors="replace").splitlines():
    m = RX.search(line)
    if m:
        rows.append(dict(s=int(m.group(1)), pred=int(m.group(2)), chosen=int(m.group(3)), tok=int(m.group(4)),
                         E=float(m.group(5)), u=float(m.group(6)), pair=float(m.group(7)),
                         uspan=float(m.group(8)), pairspan=float(m.group(9)),
                         Earg=int(m.group(10)), Uarg=int(m.group(11))))
print(f"步数={len(rows)}")
if not rows: raise SystemExit
diff = [r for r in rows if r["Earg"] != r["Uarg"]]
print(f"Earg != Uarg 的步数 = {len(diff)}/{len(rows)} = {100.0*len(diff)/len(rows):.1f}%   <-- 边项改变决定的比例")
print("按步序聚合（s: 总步数, 边项改变决定的比例, 中位 pair, 中位 uspan, 中位 pairspan, 中位 pair/uspan）:")
by = collections.defaultdict(list)
for r in rows: by[r["s"]].append(r)
for s in sorted(by)[:12]:
    g = by[s]
    ch = sum(1 for r in g if r["Earg"] != r["Uarg"])
    med = lambda k: statistics.median([r[k] for r in g])
    usp = med("uspan"); psp = med("pairspan")
    print(f"  s={s:>2} n={len(g):>3} 改变={100.0*ch/len(g):>5.1f}%  |pair|中位={statistics.median([abs(r['pair']) for r in g]):>7.4f} "
          f"uspan中位={usp:>7.4f} pairspan中位={psp:>7.4f} pairspan/uspan={psp/usp if usp else 0:>6.3f}")
allusp = statistics.median([r["uspan"] for r in rows]); allpsp = statistics.median([r["pairspan"] for r in rows])
print(f"总体: uspan中位={allusp:.4f} pairspan中位={allpsp:.4f} 比={allpsp/allusp if allusp else 0:.3f}")
print(f"|pair| 中位={statistics.median([abs(r['pair']) for r in rows]):.4f} ; pair==0 的步数="
      f"{sum(1 for r in rows if r['pair'] == 0.0)}")
PY
