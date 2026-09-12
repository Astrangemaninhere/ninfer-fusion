import json, os, re, subprocess, sys, time, urllib.request

# (A) 目标对齐的接受率对照：参考实现 KV=bfloat16 + dspark 投机，chat 协议 + 关思考，greedy
PY   = r"C:\vllm\venv\Scripts\python.exe"
J    = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
DRAFT  = os.path.join(J, "data", "draft_model")
OUT  = os.path.join(J, "dl", "vllm_spec_bf16kv.out")
ERR  = os.path.join(J, "dl", "vllm_spec_bf16kv.err")
PORT = 8144
NTOK = 7
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

spec = {"method": "dspark", "model": DRAFT, "num_speculative_tokens": NTOK,
        "draft_sample_method": "greedy"}
args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--speculative-config", json.dumps(spec),
        "--kv-cache-dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048", "--max-num-seqs", "4",
        "--enforce-eager", "--port", str(PORT), "--host", "127.0.0.1"]
print("spec = %s   KV=bfloat16" % json.dumps(spec), flush=True)
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
for i in range(240):
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
    print("=== err tail 20 ===", flush=True); print(tail(ERR, 20), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

def metrics():
    with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    acc = drf = 0.0; per = {}
    for l in [x.strip() for x in txt.splitlines() if "spec_decode" in x and not x.startswith("#")]:
        m2 = re.search(r'position="(\d+)"\}\s+([0-9.eE+-]+)', l)
        if m2: per[int(m2.group(1))] = float(m2.group(2)); continue
        m3 = re.search(r"num_accepted_tokens_total\s+([0-9.eE+-]+)", l)
        if m3: acc = float(m3.group(1)); continue
        m4 = re.search(r"num_draft_tokens_total\s+([0-9.eE+-]+)", l)
        if m4: drf = float(m4.group(1))
    return acc, drf, per

mid = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=60).read())["data"][0]["id"]
payload = {"model": mid, "messages": [{"role": "user", "content": ZH}],
           "max_tokens": 96, "temperature": 0.0, "ignore_eos": True,
           "top_p": 1.0, "seed": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
dt = time.time() - t0
msg = resp["choices"][0].get("message", {})
content = msg.get("content") or ""
a, d, per = metrics()
print("--- [spec bf16-kv] %.1fs tokens=%s/%s" % (dt, resp["usage"]["prompt_tokens"],
                                                 resp["usage"]["completion_tokens"]), flush=True)
print("    content[:110]=%r" % content[:110], flush=True)
if d > 0:
    print("    ACCEPTED=%d DRAFTED=%d RATE=%.2f%%  PER_POS=%s" %
          (a, d, 100.0*a/d, [per.get(i, 0) for i in range(NTOK)]), flush=True)
else:
    print("    (无 spec 指标)", flush=True)
print("VLLM_SPEC_BF16KV_DONE", flush=True)
try: proc.kill()
except Exception: pass
