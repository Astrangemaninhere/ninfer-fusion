#!/usr/bin/env python3
"""Workspace tidy-up, step 1: inventory only (no moves yet).

Classifies every `_*` scratch file in the working dir as
  LIVE      - currently running, or on the long-term whitelist, or invoked by a live script
  REFERENCED- named inside another script / doc / automation prompt
  ORPHAN    - none of the above (candidate for _scratch/<date>/)
Writes the inventory to _scratch/INVENTORY.md and prints the counts.
"""
import os
import pathlib
import re
import subprocess
import time

W = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
CAND_SUFFIX = (".sh", ".py", ".bat", ".ps1")

# long-term tools the user keeps running by hand / that the live pipeline needs
WHITELIST = {
    "_hf_chunked.py", "_hf_download.py", "_needle_check.sh", "_train_df2_resume.bat",
    "_train_df2_retrain.bat", "_train_resume_c.bat", "_spec_4way.sh",
    "_post_build_measure.sh", "_window_k3.sh", "_window_j2.sh", "_muse_serve_accept.sh",
    "_patchA_build.sh", "_apply_patchA.py", "_df2_shift3_probe.py", "_hit_compare.py",
    "_report_daily.sh", "_dflash2_roundtrip_tmp.py",
}
# extra live-ish helpers that other live scripts call by name
LIVE_PATTERNS = (
    "_window_", "_spec_", "_muse_serve", "_post_build", "_patchA_", "_longctx", "_sweep",
    "_df2_ab", "_df2_serve", "_train_",
)

now = time.time()
cands = sorted(p for p in W.iterdir()
               if p.is_file() and p.name.startswith("_") and p.suffix in CAND_SUFFIX)

# what is running right now?
ps = subprocess.run(["bash", "-lc", "pgrep -af 'bash|python3|powershell' || true"],
                    capture_output=True, text=True).stdout
running = set()
for line in ps.splitlines():
    for p in cands:
        if p.name in line:
            running.add(p.name)

# reference corpus: every script/bat in the workspace + our docs
corpus = []
for p in W.rglob("*"):
    if p.is_file() and (p.suffix in CAND_SUFFIX or p.suffix in (".md", ".json")):
        if "_scratch" in p.parts or "_orig_quarantine" in p.parts:
            continue
        try:
            if p.stat().st_size > 4_000_000:
                continue
            corpus.append(p)
        except OSError:
            pass

texts = {}
for p in corpus:
    try:
        texts[p] = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass

rows = []
for p in cands:
    name = p.name
    refs = []
    for q, t in texts.items():
        if q == p:
            continue
        if name in t:
            refs.append(q.relative_to(W).as_posix())
    state = "ORPHAN"
    why = ""
    if name in running:
        state = "LIVE"; why = "running now"
    elif name in WHITELIST:
        state = "LIVE"; why = "whitelist"
    elif any(pat in name for pat in LIVE_PATTERNS):
        state = "LIVE"; why = "live-pipeline pattern"
    elif refs:
        state = "REFERENCED"; why = refs[0] + (" (+%d)" % (len(refs) - 1) if len(refs) > 1 else "")

    age_h = (now - p.stat().st_mtime) / 3600.0
    rows.append((state, name, age_h, p.stat().st_size, why))

rows.sort(key=lambda r: (r[0] != "ORPHAN", -r[2]))
counts = {}
for r in rows:
    counts[r[0]] = counts.get(r[0], 0) + 1

out = ["# scratch inventory (%s)" % time.strftime("%Y-%m-%d %H:%M"),
       "",
       "| state | file | age_h | bytes | why |", "|---|---|---|---|---|"]
for st, name, age, size, why in rows:
    out.append("| %s | `%s` | %.1f | %d | %s |" % (st, name, age, size, why))
(W / "_scratch").mkdir(exist_ok=True)
(W / "_scratch" / "INVENTORY.md").write_text("\n".join(out) + "\n", encoding="utf-8")

print("candidates: %d" % len(cands))
for k in ("LIVE", "REFERENCED", "ORPHAN"):
    print("  %-10s %d" % (k, counts.get(k, 0)))
print()
print("orphans (top 25 by age):")
for st, name, age, size, why in rows:
    if st == "ORPHAN":
        print("  %-42s age=%6.1fh size=%7d" % (name, age, size))
print("... total orphans: %d" % counts.get("ORPHAN", 0))
