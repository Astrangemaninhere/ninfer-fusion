#!/usr/bin/env python3
"""Fix _hf_chunk.py: the HF API's `siblings` only carries `size` with ?blobs=true, so the
first version saw size=None and fell back to the single-GET path — the exact path that
stalls on this mirror. Ask for blobs, and fall back to a HEAD request for Content-Length."""
import pathlib

P = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_hf_chunk.py")
src = P.read_text(encoding="utf-8")

A = '''def api(endpoint: str, path: str) -> dict:
    req = urllib.request.Request(endpoint + "/api/" + path,
                                 headers={"User-Agent": "ninfer-import"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))'''
B = '''def api(endpoint: str, path: str, blobs: bool = True) -> dict:
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
    return None'''
if B.splitlines()[0] not in src:
    assert src.count(A) == 1, "api() anchor not unique"
    src = src.replace(A, B)
    print("patched: api(blobs=true) + HEAD size fallback")
else:
    print("api already patched")

C = '''        total = sizes.get(name)
        if total is None:
            total = int(fetch_range(a.endpoint, a.repo, name, 0, 0 * 0 + 1) and 0) or None'''
D = '''        total = sizes.get(name)
        if total is None:
            total = remote_size(a.endpoint, a.repo, name)
            log("   ~~ %s size from HEAD: %s" % (name, total))'''
if D not in src:
    assert src.count(C) == 1, "size fallback anchor not unique"
    src = src.replace(C, D)
    print("patched: single-GET fallback replaced by HEAD probe")
else:
    print("fallback already patched")

P.write_text(src, encoding="utf-8")
print("now %d lines" % (src.count("\n") + 1))
