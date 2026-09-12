#!/usr/bin/env python3
"""Workspace tidy-up: inventory with a SHALLOW corpus (root files + _collab + docs only).
The first version walked into models/ (200 GB) on the 9p mount and never finished."""
import os
import pathlib
import subprocess
import time

W = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
SUF = (".sh", ".py", ".bat", ".ps1")

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

# shallow corpus: root-level files + _collab/*.md|*.sh|*.py + a few docs
corpus = []
for p in W.iterdir():
    if p.is_file() and (p.suffix in SUF or p.suffix in (".md",)):
        corpus.append(p)
col = W / "_collab"
if col.is_dir():
    for p in col.iterdir():
        if p.is_file() and p.suffix in (".md", ".sh", ".py"):
            corpus.append(p)

texts = {}
for p in corpus:
    try:
        if p.stat().st_size > 3_000_000:
            continue
        texts[p] = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass

now = time.time()
rows = []
for p in cands:
    name = p.name
    refs = [q.relative_to(W).as_posix() for q, t in texts.items() if q != p and name in t]
    if name in running:
        state, why = "LIVE", "running now"
    elif name in WHITELIST:
        state, why = "LIVE", "whitelist"
    elif any(pat in name for pat in LIVE_PATTERNS):
        state, why = "LIVE", "live-pipeline pattern"
    elif refs:
        state, why = "REFERENCED", refs[0] + (" (+%d)" % (len(refs) - 1) if len(refs) > 1 else "")
    else:
        state, why = "ORPHAN", ""
    rows.append((state, name, (now - p.stat().st_mtime) / 3600.0, p.stat().st_size, why))

rows.sort(key=lambda r: (r[0] != "ORPHAN", -r[2]))
counts = {}
for r in rows:
    counts[r[0]] = counts.get(r[0], 0) + 1

out = ["# scratch inventory (%s)" % time.strftime("%Y-%m-%d %H:%M"), "",
       "corpus: %d files (root + _collab only)" % len(corpus), "",
       "| state | file | age_h | bytes | why |", "|---|---|---|---|---|"]
out += ["| %s | `%s` | %.1f | %d | %s |" % (s, n, a, z, w) for s, n, a, z, w in rows]
(W / "_scratch").mkdir(exist_ok=True)
(W / "_scratch" / "INVENTORY.md").write_text("\n".join(out) + "\n", encoding="utf-8")

print("candidates: %d" % len(cands))
for k in ("LIVE", "REFERENCED", "ORPHAN"):
    print("  %-11s %d" % (k, counts.get(k, 0)))
print()
print("orphans, newest first (up to 40):")
for s, n, a, z, w in rows:
    if s == "ORPHAN":
        print("  %-44s age=%6.1fh %7d B" % (n, a, z))
