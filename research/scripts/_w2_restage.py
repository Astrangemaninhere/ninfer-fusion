#!/usr/bin/env python3
"""窗口② 刷新：让暂存拆分产物与「已打 PART-B 的源码」一致。
规则：
  1) 覆盖型产物（_partial.cuh/_smallt.cu）以树里的幽灵副本为准（= staged + PART-B，字节账已核）。
  2) e8 兄弟 TU（不在树里）施加同一替换：static_cast<__nv_bfloat16*>(partial_acc.data) -> float*。
  3) 断言：全部暂存产物中不再有 __nv_bfloat16* 的 partial_acc 转换。
只写 _collab/ 下的暂存物，不碰源码。
"""
import hashlib, pathlib, re, sys

R = pathlib.Path("/home/user/ninfer-fusion")
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
ST = J / "_collab/build/staged"
ST2 = J / "_collab/build/staged2"

def md5(p): return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()
def lines(p): return pathlib.Path(p).read_bytes().count(b"\n")

report = []
def log(s): print(s); report.append(s)

# ---------- 1) 覆盖型：树 -> staged ----------
PAIRS = [
    ("src/ops/launcher/gqa_attention_decode_partial.cuh", ST / "gqa_attention_decode_partial.cuh.new"),
    ("src/ops/launcher/gqa_attention_decode_smallt.cu",   ST / "gqa_attention_decode_smallt.cu.new"),
]
log("== 1) 覆盖型产物：树 -> staged ==")
for tree_rel, st in PAIRS:
    tree = R / tree_rel
    if not tree.exists():
        log(f"  FAIL 树里缺 {tree_rel}"); continue
    txt = tree.read_text(errors="replace")
    if "static_cast<float*>(partial_acc.data)" not in txt and "static_cast<const float*>(partial_acc.data)" not in txt:
        log(f"  FAIL {tree_rel} 未见 FP32 partial 转换 —— 拒绝回抄"); continue
    bad = len(re.findall(r"static_cast<const?\s*__nv_bfloat16\*>\s*\(\s*partial_acc\.data\s*\)", txt))
    if bad:
        log(f"  FAIL {tree_rel} 仍有 {bad} 处 BF16 partial 转换 —— 拒绝回抄"); continue
    old = md5(st) if st.exists() else "(none)"
    st.write_bytes(tree.read_bytes())
    log(f"  ok   {st.name}: {old[:12]} -> {md5(st)[:12]}  lines={lines(st)}")

# ---------- 2) e8 兄弟 TU：施加同一替换 ----------
SUB = [
    (re.compile(r"static_cast<const\s+__nv_bfloat16\s*\*>\s*\(\s*partial_acc\.data\s*\)"),
     "static_cast<const float*>(partial_acc.data)"),
    (re.compile(r"static_cast<__nv_bfloat16\s*\*>\s*\(\s*partial_acc\.data\s*\)"),
     "static_cast<float*>(partial_acc.data)"),
]
log("== 2) e8 家族暂存产物：施加同一替换 ==")
n_tot = 0
for d in (ST, ST2):
    for p in sorted(d.glob("*.new")):
        txt = p.read_text(errors="replace")
        n = 0
        for rx, rep in SUB:
            txt, k = rx.subn(rep, txt)
            n += k
        if n:
            p.write_text(txt)
            log(f"  ok   {p.name}: {n} 处替换  md5={md5(p)[:12]} lines={lines(p)}")
            n_tot += n
log(f"  合计替换 {n_tot} 处")

# ---------- 3) 断言：无 BF16 partial 残留 ----------
log("== 3) 断言：暂存产物中无 BF16 partial 转换 ==")
left = 0
for d in (ST, ST2):
    for p in sorted(d.glob("*")):
        if p.is_file():
            k = len(re.findall(r"static_cast<const?\s*__nv_bfloat16\s*\*>\s*\(\s*partial_acc\.data\s*\)",
                               p.read_text(errors="replace")))
            if k:
                log(f"  FAIL {p.name} 仍有 {k} 处"); left += k
log("  PASS 无残留" if not left else f"  FAIL 残留 {left} 处")

# ---------- 4) 输出新的 md5/lines 表（给落地脚本用） ----------
log("== 4) 暂存产物现状（md5|lines|首行） ==")
for d in (ST, ST2):
    for p in sorted(d.glob("*")):
        if p.is_file():
            first = p.read_text(errors="replace").splitlines()[0][:60] if p.stat().st_size else ""
            log(f"  [{d.name}] {p.name}|{lines(p)}|{md5(p)}|{first}")

(J / "dl/w2_restage.log").write_text("\n".join(report) + "\n")
print("RESTAGE_LEFT=%d" % left)
sys.exit(1 if left else 0)
