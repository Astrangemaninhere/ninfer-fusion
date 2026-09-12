import json, os, re, subprocess, sys, time, urllib.request

# Windows 侧：用本地构建的 vLLM(0.26 venv) 跑“纯 greedy、无投机”，输出 token ids 供与 ninfer plain 比对
PY   = r"C:\vllm\venv\Scripts\python.exe"
J    = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
OUT  = os.path.join(J, "dl", "vllm_plain_win.out")
ERR  = os.path.join(J, "dl", "vllm_plain_win.err")
PORT = 8141
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048",
        "--max-num-seqs", "4",
        "--enforce-eager",
        "--port", str(PORT), "--host", "127.0.0.1"]
print("PY exists:", os.path.exists(PY), flush=True)
print("no speculative-config (plain greedy)", flush=True)
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)
print("server pid=%d port=%d" % (proc.pid, PORT), flush=True)

def tail(path, n=8):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""

ready = False
for i in range(200):
    time.sleep(5)
    t = tail(ERR, 12) + tail(OUT, 8)
    if re.search(r"Application startup complete|Uvicorn running on|Started server process", t):
        ready = True; break
    if re.search(r"Traceback \(most recent call last\)|ModuleNotFoundError|api_server\.py: error:", t):
        print("startup error at %ds" % (i*5), flush=True); break
    if proc.poll() is not None:
        print("server exited rc=%s at %ds" % (proc.returncode, i*5), flush=True); break
    if i % 12 == 11:
        print("  ... %ds last: %s" % ((i+1)*5, tail(ERR, 1).strip()[:110]), flush=True)
print("ready=%s" % ready, flush=True)
if not ready:
    print("=== err tail 25 ===", flush=True); print(tail(ERR, 25), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

def get(path, timeout=120):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

mid = json.loads(get("/v1/models"))["data"][0]["id"]
body = json.dumps({"model": mid, "prompt": ZH, "max_tokens": 96,
                   "temperature": 0.0, "ignore_eos": True}).encode()
req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % PORT, data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
dt = time.time() - t0
ch = resp["choices"][0]
print("--- vllm plain greedy: %.1fs finish=%s prompt_tok=%s completion_tok=%s" %
      (dt, ch.get("finish_reason"), resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]), flush=True)
print("    text[:120]=%r" % ch["text"][:120], flush=True)
# 有 token_ids 就拿（vLLM 支持 logprobs/token_ids 取决于版本；没有就靠文本比对）
ids = ch.get("token_ids")
print("    token_ids=%s" % (ids[:40] if ids else "(接口未返回)"), flush=True)
print("VLLM_PLAIN_WIN_DONE", flush=True)
try: proc.kill()
except Exception: pass
