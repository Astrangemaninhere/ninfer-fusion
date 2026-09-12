#!/usr/bin/env python3
"""用 Python 直接起 vLLM API server（subprocess 传参数列表，绕开 Windows 引号问题），
等就绪后发一次贪心请求，抓 spec-decode 指标（接受率 + 逐位置）。"""
import json, os, re, subprocess, sys, time, urllib.request

PY = r"C:\vllm\venv\Scripts\python.exe"
J = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
DRAFT = os.path.join(J, "data", "draft_model")
OUT = os.path.join(J, "dl", "vllm_ref2.out")
ERR = os.path.join(J, "dl", "vllm_ref2.err")
PORT = 8124

spec = {"method": "dspark", "model": DRAFT, "num_speculative_tokens": 7,
        "draft_sample_method": "greedy"}
spec_json = json.dumps(spec)
print("spec = %s" % spec_json, flush=True)

args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--speculative-config", spec_json,
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", str(PORT),
        "--host", "127.0.0.1"]
print("launching server ...", flush=True)
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)

def tail(path, n=12):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return "(no file)"

ready = False
for i in range(90):                      # 450s
    time.sleep(5)
    t = tail(ERR, 8)
    if re.search(r"Application startup complete|Uvicorn running|Started server", t):
        ready = True; break
    if re.search(r"Traceback \(most recent call last\)|ModuleNotFoundError|ImportError|api_server\.py: error:", t):
        print("startup error at %ds" % (i*5), flush=True); break
    if i % 6 == 5:
        print("  ... %ds" % ((i+1)*5), flush=True)
        print("    " + tail(ERR, 2).strip()[:200], flush=True)
print("ready=%s" % ready, flush=True)
if not ready:
    print("=== err tail ===", flush=True)
    print(tail(ERR, 30), flush=True)
    proc.kill()
    sys.exit(2)

print("=== greedy request ===", flush=True)
body = json.dumps({"model": TARGET, "prompt": "hello", "max_tokens": 96,
                   "temperature": 0, "ignore_eos": True}).encode()
req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=body,
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.load(r)
print("  finish=%s prompt=%s completion=%s" % (
    resp["choices"][0].get("finish_reason"), resp["usage"]["prompt_tokens"],
    resp["usage"]["completion_tokens"]), flush=True)

print("=== metrics ===", flush=True)
with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=60) as r:
    metrics = r.read().decode("utf-8", "replace")
lines = [l.strip() for l in metrics.splitlines()
         if "spec_decode" in l and not l.startswith("#")]
for l in lines[:30]:
    print("  " + l, flush=True)
acc = drf = 0.0
for l in lines:
    m = re.search(r"num_accepted_tokens[^_]*\s+([0-9.eE+-]+)", l)
    if m: acc += float(m.group(1))
    m = re.search(r"num_draft_tokens\s+([0-9.eE+-]+)", l)
    if m: drf += float(m.group(1))
print("  accepted=%s drafted=%s" % (acc, drf), flush=True)
if drf > 0:
    print("  ACCEPTANCE_RATE = %.2f%%" % (100.0*acc/drf), flush=True)
print("VLLM_REF2_DONE", flush=True)
print("(server still running on port %d; kill pid %d when done)" % (PORT, proc.pid), flush=True)
