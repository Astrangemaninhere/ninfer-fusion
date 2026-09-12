#!/usr/bin/env python3
"""干净环境（C:\\vllm\\venv-clean, vllm 0.26.0+cu132）单独跑一次 dspark 投机，抓接受率。
与 _vllm_clean.py 的差别：PY 指向 venv-clean；两档请求（hello / 西湖中文）；每档后读指标增量。
清理：先只杀"确认是 vllm api_server 且占用本端口"的进程（铁律②：先查 cmdline）。"""
import json, os, re, subprocess, sys, time, urllib.request

PY   = r"C:\vllm\venv-clean\Scripts\python.exe"
J    = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
DRAFT  = os.path.join(J, "data", "draft_model")
OUT  = os.path.join(J, "dl", "vref_clean.out")
ERR  = os.path.join(J, "dl", "vref_clean.err")
PORT = 8129
NTOK = 7
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

print("PY = %s" % PY, flush=True)
print("exists PY = %s" % os.path.exists(PY), flush=True)

# --- 端口/进程卫生：只杀占本端口的 vllm api_server ---
try:
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    pids = {m.group(1) for m in re.finditer(r":%d\s+\S+\s+LISTENING\s+(\d+)" % PORT, out)}
    for pid in pids:
        cl = subprocess.run(["wmic", "process", "where", "ProcessId=%s" % pid, "get", "CommandLine"],
                            capture_output=True, text=True).stdout
        if "vllm" in cl and "api_server" in cl:
            print("killing stale vllm api_server pid=%s" % pid, flush=True)
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        else:
            print("port %d held by non-vllm pid=%s -> 不动它" % (PORT, pid), flush=True)
except Exception as e:
    print("port hygiene skipped: %s" % e, flush=True)

spec = {"method": "dspark", "model": DRAFT, "num_speculative_tokens": NTOK,
        "draft_sample_method": "greedy"}
args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--speculative-config", json.dumps(spec),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--max-num-seqs", "32",
        "--enforce-eager",
        "--port", str(PORT), "--host", "127.0.0.1"]
print("spec = %s" % json.dumps(spec), flush=True)
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)
print("server pid = %d port %d" % (proc.pid, PORT), flush=True)

def tail(path, n=8):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""

ready = False
for i in range(180):                    # 最多 900s（含 fp4 自动调优）
    time.sleep(5)
    t = tail(ERR, 12) + tail(OUT, 8)
    if re.search(r"Application startup complete|Uvicorn running on|Started server process", t):
        ready = True; break
    if re.search(r"Traceback \(most recent call last\)|ModuleNotFoundError|api_server\.py: error:", t):
        print("startup error at %ds" % (i*5), flush=True); break
    if proc.poll() is not None:
        print("server exited early rc=%s at %ds" % (proc.returncode, i*5), flush=True); break
    if i % 12 == 11:
        print("  ... %ds last: %s" % ((i+1)*5, tail(ERR, 1).strip()[:110]), flush=True)
print("ready=%s at %s" % (ready, time.strftime("%H:%M:%S")), flush=True)
if not ready:
    print("=== err tail 25 ===", flush=True); print(tail(ERR, 25), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

def metrics():
    with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    acc = drf = 0.0; per = {}
    for l in [x.strip() for x in txt.splitlines() if "spec_decode" in x and not x.startswith("#")]:
        m = re.search(r"num_accepted_tokens_per_pos\w*\{position=", l) or re.search(r"position=\"(\d+)\"\}\s+([0-9.eE+-]+)", l)
        m2 = re.search(r"position=\"(\d+)\"\}\s+([0-9.eE+-]+)", l)
        if m2: per[int(m2.group(1))] = float(m2.group(2)); continue
        m3 = re.search(r"num_accepted_tokens_total\s+([0-9.eE+-]+)", l)
        if m3: acc = float(m3.group(1)); continue
        m4 = re.search(r"num_draft_tokens_total\s+([0-9.eE+-]+)", l)
        if m4: drf = float(m4.group(1))
    return acc, drf, per

def ask(prompt, tag):
    body = json.dumps({"model": TARGET, "prompt": prompt, "max_tokens": 96,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    dt = time.time() - t0
    ch = resp["choices"][0]
    a, d, per = metrics()
    print("--- [%s] %.1fs finish=%s prompt_tok=%s completion_tok=%s" %
          (tag, dt, ch.get("finish_reason"), resp["usage"]["prompt_tokens"],
           resp["usage"]["completion_tokens"]), flush=True)
    print("    text[:60]=%r" % ch["text"][:60], flush=True)
    if d > 0:
        print("    ACCEPTED=%d DRAFTED=%d  ACCEPTANCE_RATE=%.2f%%  PER_POS=%s" %
              (a, d, 100.0*a/d, [per.get(i, 0) for i in range(NTOK)]), flush=True)
    return a, d, per

try:
    ask("hello", "hello")
    print("=== (hello 档结束，下面换中文长 prompt) ===", flush=True)
    ask(ZH, "zh")
except Exception as e:
    print("request error: %s" % e, flush=True)
    print("=== err tail 15 ===", flush=True); print(tail(ERR, 15), flush=True)

print("=== 关键日志行 ===", flush=True)
for l in tail(OUT, 40).splitlines():
    if "SpecDecoding" in l or "Acceptance" in l or "acceptance" in l:
        print("  " + l.strip()[:200], flush=True)
print("VLLM_CLEAN_DONE", flush=True)
try: proc.kill()
except Exception: pass
