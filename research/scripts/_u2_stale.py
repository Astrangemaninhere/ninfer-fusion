#!/usr/bin/env python3
# U2: exact stale-object computation for the next build (+ optional -o/touch shortcut effect)
import os, re, sys

R = "/home/user/ninfer-fusion"
B = os.path.join(R, "build")

def mtime(p):
    try:
        return os.stat(p).st_mtime
    except OSError:
        return None

# collect all .o with their source guesses via CMake's depend files if present
objs = {}
for root, dirs, files in os.walk(B):
    if "CMakeFiles" not in root:
        continue
    for f in files:
        if f.endswith(".o"):
            objs[os.path.join(root, f)] = None

print("total .o in build:", len(objs))

# map object -> source: CMake names objects after the source file's basename
#   e.g. src/CMakeFiles/ninfer_ops.dir/ops/launcher/gqa_attention_decode.cu.o
#        -> <R>/src/ops/launcher/gqa_attention_decode.cu
def guess_src(obj):
    rel = obj.split("CMakeFiles/", 1)[1]
    rel = rel.split(".dir/", 1)[1]
    rel = rel[:-2]                      # strip .o
    cands = [os.path.join(R, "src", rel), os.path.join(R, "apps", rel)]
    for c in cands:
        if os.path.exists(c):
            return c
    # try stripping a target-specific prefix directory
    parts = rel.split("/")
    for i in range(1, min(3, len(parts)) + 1):
        c = os.path.join(R, "src", *parts[i:])
        if os.path.exists(c):
            return c
        c = os.path.join(R, "apps", *parts[i:])
        if os.path.exists(c):
            return c
    return None

stale = []
unmapped = 0
for obj in objs:
    src = guess_src(obj)
    if src is None:
        unmapped += 1
        continue
    mo, ms = mtime(obj), mtime(src)
    if mo is not None and ms is not None and ms > mo:
        stale.append((os.path.getsize(obj), obj, src, mo, ms))

stale.sort(reverse=True)
print("unmapped objects:", unmapped)
print("directly STALE objects (source newer than object):", len(stale))
print()
tot = 0
for sz, obj, src, mo, ms in stale:
    tot += sz
    print("  %10.1f MB  %s" % (sz / 1e6, obj.replace(B + "/", "")))
    print("             src=%s  [obj %s < src %s]" % (
        src.replace(R + "/", ""),
        __import__("time").strftime("%H:%M:%S", __import__("time").localtime(mo)),
        __import__("time").strftime("%H:%M:%S", __import__("time").localtime(ms))))
print()
print("sum of stale object sizes: %.1f MB" % (tot / 1e6))
