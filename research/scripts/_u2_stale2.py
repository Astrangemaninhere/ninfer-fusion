#!/usr/bin/env python3
# U2: exact stale-object set for `make ninfer ninfer-serve`
import os, time
R = "/home/user/ninfer-fusion"
B = os.path.join(R, "build")
KEEP = ("ninfer_ops", "ninfer_engine", "ninfer_serve", "ninfer_core",
        "ninfer_text", "ninfer_artifact", "ninfer_nvfp4_tma", "apps/CMakeFiles/ninfer.")

def mt(p):
    try: return os.stat(p).st_mtime
    except OSError: return None

def guess_src(obj):
    try:
        rel = obj.split("CMakeFiles/", 1)[1].split(".dir/", 1)[1]
    except IndexError:
        return None
    rel = rel[:-2]
    for base in ("src", "apps"):
        p = os.path.join(R, base, rel)
        if os.path.isfile(p): return p
    parts = rel.split("/")
    for i in range(1, min(3, len(parts)) + 1):
        for base in ("src", "apps"):
            p = os.path.join(R, base, *parts[i:])
            if os.path.isfile(p): return p
    return None

rows = []
for root, dirs, files in os.walk(B):
    if "CMakeFiles" not in root: continue
    if not any(k in root.replace("\\", "/") for k in KEEP): continue
    for f in files:
        if not f.endswith(".o"): continue
        obj = os.path.join(root, f)
        src = guess_src(obj)
        if src is None: continue
        mo, ms = mt(obj), mt(src)
        if mo and ms and ms > mo:
            rows.append((os.path.getsize(obj), obj.replace(B + "/", ""),
                         src.replace(R + "/", ""), mo, ms))

rows.sort(reverse=True)
print("== stale objects reachable from `make ninfer ninfer-serve` ==")
tot = 0
for sz, obj, src, mo, ms in rows:
    tot += sz
    print("  %8.1f MB  %-72s obj %s < src %s" % (sz/1e6, obj,
          time.strftime("%H:%M:%S", time.localtime(mo)),
          time.strftime("%H:%M:%S", time.localtime(ms))))
print("  ---- %d stale objects, %.1f MB total" % (len(rows), tot/1e6))

print()
print("== the two giants (the 40-60 min each ptxas TUs) ==")
for t in ("gqa_attention_decode.cu.o", "gqa_attention_decode_e8.cu.o"):
    hits = [o for o in rows if o[1].endswith(t)]
    print("  %-34s stale=%s  size=%.0f MB" % (t, bool(hits), hits[0][0]/1e6 if hits else -1))
