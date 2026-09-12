#!/usr/bin/env python3
"""vLLM dspark 鍙傜収娴嬮噺锛堟渶缁堢増锛夛細
 1) 鍏堟竻娈嬬暀 python锛堝惁鍒?ZMQ 绔彛鍐茬獊锛?
 2) python subprocess 璧?API server锛堢粫寮€ Windows 寮曞彿闂锛?
 3) 闀跨瓑寰咃紙fp4_gemm 鑷姩璋冧紭鍙兘瑕佸嚑鍒嗛挓锛?
 4) 璐績璇锋眰 + 鎶?spec-decode 鎺ュ彈鐜囦笌閫愪綅缃墫闈?
"""
import json, os, re, subprocess, sys, time, urllib.request

PY = r"C:\vllm\venv\Scripts\python.exe"
J = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
DRAFT = os.path.join(J, "data", "draft_model")
OUT = os.path.join(J, "dl", "vref3.out")
ERR = os.path.join(J, "dl", "vref3.err")
PORT = 8125
NTOK = 7

# --- 1) 娓呮畫鐣欙細鏀圭敱璋冪敤鏂瑰湪鍚姩鍓嶅畬鎴愶紙閬垮厤鑷潃锛?--
spec = {"method": "dspark", "model": DRAFT, "num_speculative_tokens": NTOK,
        "draft_sample_method": "greedy"}
args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
        "--port", str(PORT), "--host", "127.0.0.1"]
print("spec = %s" % json.dumps(spec), flush=True)
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)
print("server pid = %d, port %d" % (proc.pid, PORT), flush=True)

def tail(path, n=6):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""

ready = False
for i in range(180):                    # 鏈€澶?900s锛堝惈 fp4 鑷姩璋冧紭锛?
    time.sleep(5)
    t = tail(ERR, 10) + tail(OUT, 6)
    if re.search(r"Application startup complete|Uvicorn running on|Started server process", t):
        ready = True; break
    if re.search(r"Traceback \(most recent call last\)|ModuleNotFoundError|api_server\.py: error:", t):
        print("startup error at %ds" % (i*5), flush=True); break
    if proc.poll() is not None:
        print("server exited early rc=%s at %ds" % (proc.returncode, i*5), flush=True); break
    if i % 12 == 11:
        print("  ... %ds  last: %s" % ((i+1)*5, tail(ERR, 1).strip()[:110]), flush=True)

print("ready=%s" % ready, flush=True)
if not ready:
    print("=== err tail 20 ===", flush=True); print(tail(ERR, 20), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

print("=== greedy request (max_tokens=96, ignore_eos, greedy) ===", flush=True)
body = json.dumps({"model": TARGET, "prompt": "hello", "max_tokens": 96,
                   "temperature": 0.0, "ignore_eos": True}).encode()
req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=body,
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    resp = json.load(r)
ch = resp["choices"][0]
print("  finish=%s prompt=%s completion=%s" % (ch.get("finish_reason"),
      resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]), flush=True)
print("  text[:70]=%r" % ch["text"][:70], flush=True)

print("=== spec-decode metrics ===", flush=True)
with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=60) as r:
    metrics = r.read().decode("utf-8", "replace")
lines = [l.strip() for l in metrics.splitlines() if "spec_decode" in l and not l.startswith("#")]
for l in lines[:40]:
    print("  " + l, flush=True)

acc = drf = 0.0
per_pos = {}
for l in lines:
    m = re.search(r"num_accepted_tokens_per_pos\w*\{position=\"(\d+)\"\}\s+([0-9.eE+-]+)", l)
    if m: per_pos[int(m.group(1))] = float(m.group(2)); continue
    m = re.search(r"num_accepted_tokens_total\s+([0-9.eE+-]+)", l)
    if m: acc = float(m.group(1)); continue
    m = re.search(r"num_draft_tokens_total\s+([0-9.eE+-]+)", l)
    if m: drf = float(m.group(1))
print("  accepted=%s drafted=%s" % (acc, drf), flush=True)
if drf > 0:
    print("  ACCEPTANCE_RATE = %.2f%%" % (100.0*acc/drf), flush=True)
if per_pos:
    print("  PER_POSITION(counts) = %s" % [per_pos.get(i, 0) for i in range(NTOK)], flush=True)
print("VLLM_REF3_DONE", flush=True)
print("(server alive: pid %d port %d 鈥?kill when done)" % (proc.pid, PORT), flush=True)
