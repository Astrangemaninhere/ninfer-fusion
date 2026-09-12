import json, os, re, subprocess, sys, time, urllib.request

# 协议对齐版：chat/completions + enable_thinking=false（对应 ninfer 的 --no-thinking），纯 greedy
PY   = r"C:\vllm\venv\Scripts\python.exe"
J    = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
OUT  = os.path.join(J, "dl", "vllm_chat_win.out")
ERR  = os.path.join(J, "dl", "vllm_chat_win.err")
PORT = 8142
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048", "--max-num-seqs", "4",
        "--enforce-eager", "--port", str(PORT), "--host", "127.0.0.1"]
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)
print("server pid=%d port=%d (chat/completions + enable_thinking=false)" % (proc.pid, PORT), flush=True)

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
print("ready=%s" % ready, flush=True)
if not ready:
    print("=== err tail 20 ===", flush=True); print(tail(ERR, 20), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

def post(path, payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

mid = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=60).read())["data"][0]["id"]

# 关键：chat 协议 + 关思考（Qwen3.5 模板的 enable_thinking）
for tag, extra in (("thinking_off", {"chat_template_kwargs": {"enable_thinking": False}}),
                   ("thinking_default", {})):
    payload = {"model": mid, "messages": [{"role": "user", "content": ZH}],
               "max_tokens": 96, "temperature": 0.0, "ignore_eos": True,
               "top_p": 1.0, "seed": 0}
    payload.update(extra)
    try:
        t0 = time.time()
        resp = post("/v1/chat/completions", payload)
        dt = time.time() - t0
        ch = resp["choices"][0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        print("--- [%s] %.1fs finish=%s prompt_tok=%s completion_tok=%s" %
              (tag, dt, ch.get("finish_reason"), resp["usage"]["prompt_tokens"],
               resp["usage"]["completion_tokens"]), flush=True)
        print("    content[:150]=%r" % content[:150], flush=True)
        print("    reasoning[:80]=%r" % reasoning[:80], flush=True)
    except Exception as e:
        print("--- [%s] error: %s %s" % (tag, type(e).__name__, str(e)[:200]), flush=True)

print("VLLM_CHAT_WIN_DONE", flush=True)
try: proc.kill()
except Exception: pass
