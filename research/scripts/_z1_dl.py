#!/usr/bin/env python3
"""Z1: resumable downloader from hf-mirror for a chosen repo's needed files."""
import os, sys, time, urllib.request

ENDPOINT = "https://hf-mirror.com"

def fetch(repo, rfilename, dest_dir, retries=6):
    dest = os.path.join(dest_dir, rfilename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = "%s/%s/resolve/main/%s" % (ENDPOINT, repo, rfilename)
    for attempt in range(retries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                total = r.headers.get("Content-Length")
                total = int(total) + have if total else None
                code = r.status
                if have and code != 206:
                    # server ignored Range -> restart
                    have = 0
                mode = "ab" if have else "wb"
                t0 = time.time()
                with open(dest, mode) as f:
                    n = have
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        n += len(chunk)
                        el = time.time() - t0
                        if el > 3 and (n - have) > 0:
                            pct = (n / total * 100) if total else 0
                            print("  %-40s %6.2f/%.2f GB  %5.1f MB/s" %
                                  (rfilename[:40], n / 2**30,
                                   (total or 0) / 2**30, n / 2**20 / el),
                                  flush=True)
        except Exception as e:
            sys.stderr.write("  retry %d for %s: %s\n" % (attempt, rfilename, e))
            time.sleep(3)
            continue
        if total is None or os.path.getsize(dest) >= total:
            print("  DONE %s (%.3f GiB)" % (rfilename, os.path.getsize(dest) / 2**30), flush=True)
            return True
    return False

if __name__ == "__main__":
    repo = sys.argv[1]
    dest_dir = sys.argv[2]
    files = sys.argv[3:]
    ok = True
    for f in files:
        print("== %s" % f, flush=True)
        if not fetch(repo, f, dest_dir):
            ok = False
            print("  FAILED %s" % f, flush=True)
    print("ALL_OK" if ok else "SOME_FAILED", flush=True)
