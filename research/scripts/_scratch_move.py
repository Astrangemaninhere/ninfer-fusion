#!/usr/bin/env python3
"""Workspace tidy-up, step 2: move ORPHAN scratch scripts into _scratch/<date>/.
Re-runs the same classification (no side effects on LIVE/REFERENCED), then writes the
index and the whitelist. Nothing referenced, running, or whitelisted is touched."""
import pathlib
import shutil
import subprocess
import time

W = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
SUF = (".sh", ".py", ".bat", ".ps1")
S = W / "_scratch"

WHITELIST = {
    "_hf_chunked.py", "_hf_download.py", "_needle_check.sh", "_train_df2_resume.bat",
    "_train_df2_retrain.bat", "_train_resume_c.bat", "_spec_4way.sh",
    "_post_build_measure.sh", "_window_k3.sh", "_muse_serve_accept.sh",
    "_patchA_build.sh", "_apply_patchA.py", "_df2_shift3_probe.py",
}
LIVE_PATTERNS = ("_window_", "_spec_", "_muse_serve", "_post_build", "_patchA_", "_train_", "_longctx")

cands = sorted(p for p in W.iterdir() if p.is_file() and p.name.startswith("_") and p.suffix in SUF)
ps = subprocess.run(["bash", "-lc", "pgrep -af 'bash|python3|powershell' || true"],
                    capture_output=True, text=True).stdout
running = {p.name for p in cands if p.name in ps}

corpus = [p for p in W.iterdir() if p.is_file() and p.suffix in SUF + (".md",)]
col = W / "_collab"
if col.is_dir():
    corpus += [p for p in col.iterdir() if p.is_file() and p.suffix in (".md", ".sh", ".py")]
texts = {}
for p in corpus:
    try:
        if p.stat().st_size <= 3_000_000:
            texts[p] = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass

def classify(p):
    name = p.name
    if name in running:
        return "LIVE", "running now"
    if name in WHITELIST:
        return "LIVE", "whitelist"
    if any(pat in name for pat in LIVE_PATTERNS):
        return "LIVE", "live-pipeline pattern"
    refs = [q.relative_to(W).as_posix() for q, t in texts.items() if q != p and name in t]
    if refs:
        return "REFERENCED", refs[0] + (" (+%d)" % (len(refs) - 1) if len(refs) > 1 else "")
    return "ORPHAN", ""

moved, live, refd = [], [], []
for p in cands:
    state, why = classify(p)
    if state != "ORPHAN":
        (live if state == "LIVE" else refd).append((p.name, why))
        continue
    day = time.strftime("%Y-%m-%d", time.localtime(p.stat().st_mtime))
    dest = S / day
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(p), str(dest / p.name))
    moved.append((day, p.name))

print("moved %d orphan(s) into _scratch/<date>/" % len(moved))
print("left in place: %d live, %d referenced" % (len(live), len(refd)))

# index by theme keyword
themes = {
    "构建/编译": ("build", "make", "win", "j2", "k2", "k3", "patch"),
    "测量/接受率": ("ab", "acc", "spec", "dflash", "df2", "mtp", "dspark", "hist", "pos", "curve"),
    "KV/量化": ("kv", "e8", "iso3", "fp8", "nvfp4", "i8", "cold"),
    "导出/产物": ("export", "tuned", "artifact", "patch_dflash", "verify", "roundtrip"),
    "训练": ("train", "ckpt", "w9", "cache", "hs"),
    "下载/导入": ("hf", "download", "chunk", "import", "adapt", "archkit"),
    "长上下文/质量": ("needle", "long", "quality", "ppl", "perplexity", "code"),
    "Muse": ("muse", "glimmer", "nan"),
}
lines = ["# _scratch index", "",
         "Moved here by `_scratch_inventory2.py` + `_scratch_move.py` on %s." % time.strftime("%Y-%m-%d %H:%M"),
         "Rule: not running, not whitelisted, and its name appears in no script/doc — see `INVENTORY.md`",
         "for the per-file evidence and `WHITELIST.md` for what deliberately stayed at the root.", ""]
dates = sorted({d for d, _ in moved})
for d in dates:
    files = sorted(n for dd, n in moved if dd == d)
    lines += ["## %s (%d files)" % (d, len(files))]
    used = set()
    for theme, keys in themes.items():
        hits = [f for f in files if any(k in f.lower() for k in keys) and f not in used]
        if hits:
            used.update(hits)
            lines.append("- **%s**: %s" % (theme, ", ".join("`%s`" % h for h in hits[:14])
                                           + (" …" if len(hits) > 14 else "")))
    rest = [f for f in files if f not in used]
    if rest:
        lines.append("- 其他: %s" % ", ".join("`%s`" % r for r in rest[:14]) + (" …" if len(rest) > 14 else ""))
    lines.append("")
(S / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

wl = ["# 长期工具白名单（留在工作区根目录，勿移）", "",
      "这些是手工在跑或活管线按绝对路径调用的；`_scratch_move.py` 每次都会跳过它们。", ""]
for n, why in sorted(live):
    wl.append("- `%s` — %s" % (n, why))
wl += ["", "## 被脚本/文档引用而保留在根目录的（%d 个）" % len(refd), ""]
for n, why in sorted(refd)[:40]:
    wl.append("- `%s` — 被 %s 引用" % (n, why))
if len(refd) > 40:
    wl.append("- … 其余 %d 个见 `INVENTORY.md` 的 REFERENCED 行" % (len(refd) - 40))
(S / "WHITELIST.md").write_text("\n".join(wl) + "\n", encoding="utf-8")
print("wrote _scratch/README.md and _scratch/WHITELIST.md")
