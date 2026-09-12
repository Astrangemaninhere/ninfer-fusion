#!/usr/bin/env python3
"""Chunked downloader for the two requested models.

Full-file GETs against the mirror stall on large blobs (measured: the 2.16 GB
MiniCPM safetensors froze at 479 MB), while range GETs streamed 200 MB in 9 s.
So each file is fetched in 100 MB ranges, appended to the target with resume.
"""
import json
import os
import subprocess
import time
import urllib.request

ENDPOINT = "https://hf-mirror.com"
BASE = r"C:\Users\User\Documents\ziqinzhang\models"
LOG = r"C:\Users\User\Documents\ziqinzhang\dl\hf-chunk.log"
TMP = r"C:\Users\User\Documents\ziqinzhang\dl\_chunk.tmp"
CHUNK = 100 * 1024 * 1024

JOBS = [
    ("openbmb/MiniCPM5-1B", "MiniCPM5-1B"),
    ("dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4", "Qwen3.8-Flash-Next-ABLITERATED-NVFP4"),
]


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def api(path):
    req = urllib.request.Request(ENDPOINT + "/api/" + path,
                                 headers={"User-Agent": "ninfer-dl"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch(url, path, total, label):
    """Append ranges until `total` bytes are present."""
    done = os.path.getsize(path) if os.path.exists(path) else 0
    if done > total:                       # corrupt over-long file: restart it
        os.remove(path)
        done = 0
    while done < total:
        end = min(done + CHUNK, total) - 1
        rc = subprocess.call(["curl.exe", "-sL", "--retry", "4", "--retry-delay", "2",
                              "-r", "%d-%d" % (done, end), url, "-o", TMP])
        got = os.path.getsize(TMP) if os.path.exists(TMP) else 0
        expect = end - done + 1
        if rc != 0 or got != expect:
            log("   !! %s chunk %d-%d rc=%d got=%d/%d, retrying" %
                (label, done, end, rc, got, expect))
            time.sleep(2)
            continue
        with open(TMP, "rb") as src, open(path, "ab") as dst:
            dst.write(src.read())
        done += expect
        if done % (500 * 1024 * 1024) < CHUNK:
            log("   .. %s %.2f/%.2f GB" % (label, done / 1e9, total / 1e9))
    if os.path.exists(TMP):
        os.remove(TMP)
    return done


for repo, folder in JOBS:
    dest = os.path.join(BASE, folder)
    os.makedirs(dest, exist_ok=True)
    meta = api("models/" + repo + "?blobs=true")
    entries = [(s["rfilename"], s.get("size") or 0) for s in meta.get("siblings", [])
               if not s["rfilename"].startswith(".")]
    log("=== %s: %d files, %.1f GB" % (repo, len(entries), sum(e[1] for e in entries) / 1e9))
    for name, size in entries:
        path = os.path.join(dest, name.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = "%s/%s/resolve/main/%s" % (ENDPOINT, repo, name)
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == size and size > 0:
            continue
        if size == 0:
            subprocess.call(["curl.exe", "-sL", url, "-o", path])
            log("   [ok] %s" % name)
            continue
        t0 = time.time()
        got = fetch(url, path, size, name[:40])
        log("   [ok] %-58s %.1f MB in %.0fs" % (name[:58], got / 1e6, time.time() - t0))
log("ALL_CHUNKED_DOWNLOADS_FINISHED")
