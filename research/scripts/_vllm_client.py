#!/usr/bin/env python3
"""对已起的 spec 服务器(8126)发贪心请求并抓 spec-decode 接受率与逐位置剖面。"""
import json, re, urllib.request, sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8126
TARGET = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-NVFP4-RTX5090"
BASE = "http://127.0.0.1:%d" % PORT

print("=== 请求（max_tokens=96, greedy, ignore_eos）===", flush=True)
body = json.dumps({"model": TARGET, "prompt": "hello", "max_tokens": 96,
                   "temperature": 0.0, "ignore_eos": True}).encode()
req = urllib.request.Request(BASE + "/v1/completions", data=body,
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    resp = json.load(r)
ch = resp["choices"][0]
print("  finish=%s prompt=%s completion=%s" % (ch.get("finish_reason"),
      resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]), flush=True)
print("  text[:80]=%r" % ch["text"][:80], flush=True)

print("=== spec-decode 指标 ===", flush=True)
with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
    metrics = r.read().decode("utf-8", "replace")
lines = [l.strip() for l in metrics.splitlines() if "spec_decode" in l and not l.startswith("#")]
for l in lines[:40]:
    print("  " + l, flush=True)

acc = drf = 0.0
per_pos = {}
for l in lines:
    m = re.search(r'num_accepted_tokens_per_pos\w*\{position="(\d+)"\}\s+([0-9.eE+-]+)', l)
    if m:
        per_pos[int(m.group(1))] = float(m.group(2)); continue
    m = re.search(r"num_accepted_tokens_total\s+([0-9.eE+-]+)", l)
    if m:
        acc = float(m.group(1)); continue
    m = re.search(r"num_draft_tokens_total\s+([0-9.eE+-]+)", l)
    if m:
        drf = float(m.group(1))
print()
print("  accepted=%s  drafted=%s" % (acc, drf), flush=True)
if drf > 0:
    print("  === ACCEPTANCE_RATE = %.2f%% ===" % (100.0*acc/drf), flush=True)
if per_pos:
    print("  PER_POSITION(counts) = %s" % [per_pos.get(i, 0) for i in range(9)], flush=True)
print("CLIENT_DONE", flush=True)
