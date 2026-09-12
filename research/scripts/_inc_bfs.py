#!/usr/bin/env python3
"""BFS 求 4 个头文件的传递包含者（落到 .cpp/.cu 编译单元），只读不动树。"""
import pathlib, re, collections
R = pathlib.Path("/home/user/ninfer-fusion")
SEEDS = ["layouts_impl.h", "program_impl.h", "kv_calibration.h", "text_context_impl.h"]

# 建索引：basename -> [包含它的文件]
idx = collections.defaultdict(list)
TU_SUFFIX = (".cpp", ".cu", ".cc")
allfiles = []
for root in (R / "src", R / "apps", R / "include"):
    for p in root.rglob("*"):
        if p.suffix in (".h", ".cuh", ".hpp", ".cpp", ".cu", ".cc") and ".orig" not in p.name:
            allfiles.append(p)
for p in allfiles:
    try: txt = p.read_text(errors="replace")
    except Exception: continue
    for s in SEEDS + []:
        if s in txt:
            idx[s].append(p)

def bfs(seed):
    seen, tus = set(), set()
    stack = [p for p in idx.get(seed, []) if p.name != seed]
    while stack:
        p = stack.pop()
        if p in seen: continue
        seen.add(p)
        if p.suffix in TU_SUFFIX:
            tus.add(p); continue
        try: txt = p.read_text(errors="replace")
        except Exception: continue
        for q in allfiles:
            if q.name != p.name and p.name in q.read_text(errors="replace")[:200000]:
                if q not in seen: stack.append(q)
    return sorted(seen), sorted(tus)

for s in SEEDS:
    hdr, tus = bfs(s)
    print(f"== {s}: 中间头 {len(hdr)} 个, 编译单元 {len(tus)} 个")
    for t in tus[:12]:
        print(f"     {t.relative_to(R)}")
    if len(tus) > 12: print(f"     ... 其余 {len(tus)-12} 个")

# 构建进度
import subprocess, time
print("=== 编译进度 ===")
print(subprocess.run(["bash","-lc","ps -eo etimes,args | grep -E 'make ninfer|nvcc' | grep -v grep | cut -c1-70 | head -5"],
                     capture_output=True, text=True).stdout)
log = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl/build_split.log")
print("--- 日志尾 ---")
print("\n".join(log.read_text(errors="replace").splitlines()[-12:]))
