#!/usr/bin/env python3
"""Fetch just the metadata of XHToken/Spark-X2.5-4B through the Windows-side proxy.

The importer's S1-S3 stages (identity, spec extraction, gap classification) only need
config.json / generation_config.json / tokenizer_config.json / the safetensors index —
a few hundred KB. Weights come later, and only if the spec stage says the arch is supported.

Mirror first (hf-mirror.com) because the earlier session measured that the main hub stalls
on large blobs while the mirror served range GETs at ~20 MB/s.
"""
import json
import os
import sys
import urllib.request

PROXY = "http://127.0.0.1:10808"
REPO = sys.argv[1] if len(sys.argv) > 1 else "XHToken/Spark-X2.5-4B"
DEST = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\User\Documents\ziqinzhang\models\Spark-X2.5-4B"
ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
urllib.request.install_opener(opener)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "ninfer-import"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


os.makedirs(DEST, exist_ok=True)
files = json.loads(get(ENDPOINTS[0] + "/api/models/" + REPO).decode("utf-8"))["siblings"]
names = [f["rfilename"] for f in files]
print("%s: %d files" % (REPO, len(names)))
small = [n for n in names if n.endswith((".json", ".txt", ".md", ".py", ".model"))]
print("metadata files: %d" % len(small))

ok = 0
for n in small:
    out = os.path.join(DEST, n.replace("/", os.sep))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("  skip (have) %s" % n); ok += 1; continue
    for ep in ENDPOINTS:
        try:
            data = get("%s/%s/resolve/main/%s" % (ep, REPO, n), timeout=120)
            with open(out, "wb") as f:
                f.write(data)
            print("  %-46s %8d B  (%s)" % (n, len(data), ep.split("//")[1]))
            ok += 1
            break
        except Exception as e:
            print("  FAIL %s via %s: %s" % (n, ep.split("//")[1], str(e)[:70]))
print("fetched %d/%d" % (ok, len(small)))
big = [n for n in names if n.endswith(".safetensors")]
print("weights files not fetched (%d): %s" % (len(big), ", ".join(big[:4])))
