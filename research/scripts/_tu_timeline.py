#!/usr/bin/env python3
"""从对象文件时间线反推每个 TU 的真实耗时（构建是 -j1 串行 => mtime 差 = 耗时）。

为什么要反推：make 不打印每个文件的耗时，而"哪个 TU 是长杆"决定了拆谁。
串行构建下，第 i 个 .o 的 mtime 减去前一个 .o 的 mtime 就是第 i 个 TU 的编译时长；
超过 1h 的空档视为"构建暂停/我手动停过"，排除掉。
"""
import pathlib
import subprocess
import sys
from datetime import datetime

B = pathlib.Path("/home/user/ninfer-fusion/build")
rows = []
for p in B.rglob("*.o"):
    try:
        rows.append((p.stat().st_mtime, str(p.relative_to(B))))
    except OSError:
        pass
rows.sort()
if not rows:
    print("没有 .o 文件")
    sys.exit(1)

# 只取时间上连续的那一段（最后一段）
gaps = [(rows[i][0] - rows[i - 1][0], i) for i in range(1, len(rows))]
big = [g for g, _ in gaps if g > 3600]
start = 0
if big:
    last_idle = max(i for g, i in gaps if g > 3600)
    start = last_idle
seg = rows[start:]
total = seg[-1][0] - seg[0][0]
print("=== 时间线 ===")
print("段内对象数: %d  段跨度: %.1f 分钟 (%s .. %s)"
      % (len(seg), total / 60.0,
         datetime.fromtimestamp(seg[0][0]).strftime("%H:%M:%S"),
         datetime.fromtimestamp(seg[-1][0]).strftime("%H:%M:%S")))
print("(跳过 %d 段，%d 个超过 1h 的空档 = 构建暂停)" % (start, len(big)))

deltas = []
for i in range(1, len(seg)):
    deltas.append((seg[i][0] - seg[i - 1][0], seg[i][1], seg[i][0]))
deltas.sort(reverse=True)
print()
print("=== 单 TU 耗时 Top 20（秒）===")
acc = 0.0
for d, name, mt in deltas[:20]:
    acc += d
    print("  %7.1fs  %s  %s" % (d, datetime.fromtimestamp(mt).strftime("%H:%M:%S"), name))
print()
print("Top20 合计: %.1f 分钟 / 段跨度 %.1f 分钟 = %.0f%%"
      % (acc / 60, total / 60, 100.0 * acc / total))
# 分类汇总：按顶层目录
from collections import defaultdict
agg = defaultdict(float)
for d, name, _ in deltas:
    parts = name.split("/")
    key = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:2])
    agg[key] += d
print()
print("=== 按目录汇总（前 12）===")
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
    print("  %7.1f 分钟  %s" % (v / 60, k))

print()
print("=== 机器 ===")
print(subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip(), "cores")
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.strip())
print(subprocess.run(["bash", "-lc", "swapon --show 2>/dev/null | head -3 || echo '(no swap)'"],
                     capture_output=True, text=True).stdout.strip())
