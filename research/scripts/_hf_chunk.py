#!/usr/bin/env python3
"""_hf_chunk.py — generic resumable chunked fetcher for HF mirrors.

Why chunked: measured in this project, plain full-file GETs against the mirror stall on
large blobs, while 100 MB range GETs stream steadily; each file is therefore fetched as a
sequence of ranges appended to the target with resume, so a killed run loses at most one
chunk.

Usage:
  python _hf_chunk.py <repo> <dest_dir> [--files a.safetensors,b.json] [--endpoint URL]
                      [--chunk-mb 100] [--include-suffix .safetensors]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

PROXY = "http://127.0.0.1:10808"
LOG = r"C:\Users\User\Documents\ziqinzhang\dl\hf-chunk.log"


def log(msg: str) -> None:
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))


def api(endpoint: str, path: str, blobs: bool = True) -> dict:
    # blobs=true is what makes each sibling carry `size`; without it every file looks
    # sizeless and the caller falls back to a full-file GET, which stalls on this mirror.
    url = endpoint + "/api/" + path + ("?blobs=true" if blobs else "")
    req = urllib.request.Request(url, headers={"User-Agent": "ninfer-import"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def remote_size(ep: str, repo: str, name: str) -> int | None:
    """Content-Length from a HEAD; used when the API still does not report a size."""
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(
                "%s/%s/resolve/main/%s" % (ep, repo, name),
                headers={"User-Agent": "ninfer-import", "Range": "bytes=0-0"},
                method=method)
            with opener().open(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
                cl = r.headers.get("Content-Length")
                if cl:
                    return int(cl)
        except Exception:
            continue
    return None


def fetch_range(ep: str, repo: str, name: str, lo: int, hi: int, retries: int = 6) -> bytes:
    url = "%s/%s/resolve/main/%s" % (ep, repo, name)
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ninfer-import", "Range": "bytes=%d-%d" % (lo, hi)})
            with opener().open(req, timeout=300) as r:
                data = r.read()
            if len(data) != hi - lo + 1:
                raise IOError("short read %d/%d" % (len(data), hi - lo + 1))
            return data
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last = e
            log("   !! %s chunk %d-%d rc=%s retry %d/%d" % (name, lo, hi, type(e).__name__,
                                                            attempt, retries))
            time.sleep(2 * attempt)
    raise RuntimeError("chunk %s %d-%d failed: %s" % (name, lo, hi, last))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("dest")
    ap.add_argument("--files", default="", help="comma separated; default = all siblings")
    ap.add_argument("--include-suffix", default="", help="e.g. .safetensors")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--chunk-mb", type=int, default=100)
    a = ap.parse_args()

    chunk = a.chunk_mb * 1024 * 1024
    info = api(a.endpoint, "models/" + a.repo)
    sizes = {f["rfilename"]: f.get("size") for f in info["siblings"]}
    if a.files:
        names = [n.strip() for n in a.files.split(",") if n.strip()]
    else:
        names = [n for n in sizes if not a.include_suffix or n.endswith(a.include_suffix)]
    os.makedirs(a.dest, exist_ok=True)
    log("=== %s -> %s : %d file(s) via %s" % (a.repo, a.dest, len(names), a.endpoint))

    for name in names:
        out = os.path.join(a.dest, name.replace("/", os.sep))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        total = sizes.get(name)
        if total is None:
            total = remote_size(a.endpoint, a.repo, name)
            log("   ~~ %s size from HEAD: %s" % (name, total))
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if total and have >= total:
            log("   == %s already complete (%d B)" % (name, have))
            continue
        if total is None:  # size unknown: fall back to a single GET
            log("   ?? %s size unknown -> single GET" % name)
            open(out, "wb").write(fetch_range(a.endpoint, a.repo, name, 0, 2**31 - 1))
            log("   [ok] %s" % name)
            continue
        log("   .. %s %.2f/%.2f GB" % (name, have / 1e9, total / 1e9))
        t0 = time.time()
        with open(out, "ab") as f:
            while have < total:
                hi = min(have + chunk, total) - 1
                f.write(fetch_range(a.endpoint, a.repo, name, have, hi))
                have = hi + 1
                if (have // chunk) % 5 == 0:
                    rate = have / max(1e-6, time.time() - t0) / 1e6
                    log("   .. %s %.2f/%.2f GB (%.1f MB/s)" % (name, have / 1e9,
                                                              total / 1e9, rate))
        log("   [ok] %s %.1f MB in %ds" % (name, total / 1e6, int(time.time() - t0)))
    log("ALL_CHUNKED_DOWNLOADS_FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
