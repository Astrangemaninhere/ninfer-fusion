#!/usr/bin/env python3
"""Download the two requested models through the hf-mirror endpoint.

 1. openbmb/MiniCPM5-1B                                (2.2 GB, auto-import target)
 2. dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4     (135 GB, abliterated + NVFP4 + PLE)

Measured: mirror 21.6 MB/s direct, huggingface.co via the local proxy only 9.8 MB/s
(and the hub's default xet path was ~33 KB/s), so HF_ENDPOINT points at the mirror and
the proxy env is cleared for this process.
Resumable: complete files are skipped, partial ones continue.
"""
import os
import time

for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
    os.environ.pop(var, None)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# The hub defaults to the xet chunked protocol on modern versions; on this link xet
# stalls (~33 KB/s with two pm_*.incomplete parts) while plain HTTP over the mirror
# measured 21.6 MB/s. Force the plain path.
os.environ["HF_HUB_DISABLE_XET"] = "1"

BASE = r"C:\Users\User\Documents\ziqinzhang\models"
LOG = r"C:\Users\User\Documents\ziqinzhang\dl\hf-download.log"

from huggingface_hub import snapshot_download  # noqa: E402


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


JOBS = [
    ("openbmb/MiniCPM5-1B", os.path.join(BASE, "MiniCPM5-1B")),
    ("dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4",
     os.path.join(BASE, "Qwen3.8-Flash-Next-ABLITERATED-NVFP4")),
]

for repo, dest in JOBS:
    log("=== start %s -> %s (endpoint=%s)" % (repo, dest, os.environ["HF_ENDPOINT"]))
    t0 = time.time()
    try:
        path = snapshot_download(repo_id=repo, local_dir=dest, max_workers=8)
        log("=== done %s in %.1f min -> %s" % (repo, (time.time() - t0) / 60.0, path))
    except Exception as exc:
        log("=== FAIL %s: %s" % (repo, str(exc)[:200]))
log("ALL_DOWNLOADS_FINISHED")
