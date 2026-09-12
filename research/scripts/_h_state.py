#!/usr/bin/env python3
"""精确判定每个已备补丁的真实应用态：按文件统计"新增行"在树中是否已存在。
不猜、不看板，只看字节。"""
import pathlib, re

R = pathlib.Path("/home/user/ninfer-fusion")
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab")
PATCHES = ["A_n1_patch.diff", "A_n1b_cold_pages.diff", "A_s30_budget_cold.diff",
           "A_s24_window_table.diff", "B_s33_ple_wiring.diff"]

def crlf(p):
    b = p.read_bytes()
    return b.count(b"\r\n")

for name in PATCHES:
    f = J / name
    if not f.exists():
        print(f"== {name}: 缺文件"); continue
    txt = f.read_text(errors="replace").splitlines()
    cur = None
    per = {}
    for ln in txt:
        if ln.startswith("+++ b/"):
            cur = ln[6:].strip(); per.setdefault(cur, {"add": [], "del": []}); continue
        if cur is None or ln.startswith(("+++", "---", "@@")):
            continue
        if ln.startswith("+"): per[cur]["add"].append(ln[1:].strip())
        elif ln.startswith("-"): per[cur]["del"].append(ln[1:].strip())
    print(f"== {name}")
    for path, d in per.items():
        t = R / path
        if not t.exists():
            print(f"   {path}: 树中无此文件  (+{len(d['add'])}/-{len(d['del'])})"); continue
        body = t.read_text(errors="replace")
        add = [a for a in d["add"] if a]
        hit = sum(1 for a in add if a in body)
        strikes = [d0 for d0 in d["del"] if d0 and d0 in body]
        if add:
            if hit == len(add): state = "**已应用**"
            elif hit == 0:      state = "未应用"
            else:               state = f"**部分({hit}/{len(add)})**"
        else:
            state = "无新增行(纯删除)"
        print(f"   {path}: +{len(add)}/-{len(d['del'])} 新增行命中={hit}  删除行仍在={len(strikes)}  {state}"
              f"{'  [CRLF:%d]' % crlf(t) if crlf(t) else ''}")
    print()
