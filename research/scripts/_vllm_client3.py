#!/usr/bin/env python3
"""Measure vLLM spec-decode acceptance (ASCII only). Usage: client3.py <port>"""
import json, re, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8127
TARGET = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-NVFP4-RTX5090"
P = ("\u8bf7\u7528\u4e2d\u6587\u5199\u4e00\u6bb5\u4e24\u767e\u5b57\u5de6\u53f3\u7684\u77ed\u6587\uff0c"
     "\u4ecb\u7ecd\u897f\u6e56\u4e00\u5e74\u56db\u5b63\u7684\u666f\u8272\u53d8\u5316\uff0c"
     "\u8981\u6c42\u8bed\u53e5\u8fde\u8d2f\u3001\u4e0d\u8981\u5217\u6761\u76ee\u3002")
BASE = "http://127.0.0.1:%d" % PORT

def snap():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
        m = r.read().decode("utf-8", "replace")
    acc = drf = dr = 0.0
    per = {}
    for l in m.splitlines():
        if l.startswith("#"):
            continue
        for key, pat in (("acc", r"spec_decode_num_accepted_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)"),
                         ("drf", r"spec_decode_num_draft_tokens_total\{[^}]*\}\s+([0-9.eE+-]+)"),
                         ("dr",  r"spec_decode_num_drafts_total\{[^}]*\}\s+([0-9.eE+-]+)")):
            mm = re.search(pat, l)
            if mm:
                v = float(mm.group(1))
                if key == "acc": acc = v
                elif key == "drf": drf = v
                else: dr = v
        mm = re.search(r'spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([0-9.eE+-]+)', l)
        if mm:
            per[int(mm.group(1))] = float(mm.group(2))
    return acc, drf, dr, per

a0, d0, r0, p0 = snap()
print("=== request (same prompt as ninfer: Chinese, greedy, 96 tokens) ===", flush=True)
body = json.dumps({"model": TARGET, "prompt": P, "max_tokens": 96,
                   "temperature": 0.0, "ignore_eos": True}).encode()
req = urllib.request.Request(BASE + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
ch = resp["choices"][0]
print("  finish=%s prompt_tokens=%s completion_tokens=%s" % (
    ch.get("finish_reason"), resp["usage"]["prompt_tokens"],
    resp["usage"]["completion_tokens"]), flush=True)

a1, d1, r1, p1 = snap()
da, dd, dr = a1 - a0, d1 - d0, r1 - r0
dper = {k: int(p1.get(k, 0) - p0.get(k, 0)) for k in set(list(p0) + list(p1))}
print()
print("=== DELTA (this request) ===", flush=True)
print("  rounds=%d  draft_tokens=%d  accepted=%d" % (dr, dd, da), flush=True)
if dd > 0:
    print("  === ACCEPTANCE = %.2f%% ===" % (100.0 * da / dd), flush=True)
    print("  acceptance_length = %.3f" % (1.0 + da / max(1.0, dr)), flush=True)
print("  PER_POSITION(counts) = %s" % [dper.get(i, 0) for i in range(9)], flush=True)
print("  PER_POSITION(rate)   = %s" % [
    ("%.1f%%" % (100.0 * dper.get(i, 0) / dr)) if dr else "-" for i in range(7)], flush=True)
print("  (ref healthy N=7: 62.6%/34.9%/19.1%/9.6%/4.7%/1.5%/0.6%; ninfer: 24%/1.4%/0/0/0/0/0)",
      flush=True)
print("CLIENT3_DONE", flush=True)
