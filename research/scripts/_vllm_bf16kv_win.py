import json, os, re, subprocess, sys, time, urllib.request

# 共同分母：bf16 KV（vLLM 侧显式指定），chat 协议 + enable_thinking=false，纯 greedy
PY   = r"C:\vllm\venv\Scripts\python.exe"
J    = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(J, "models", "Qwen3.8-27B-NVFP4-RTX5090")
OUT  = os.path.join(J, "dl", "vllm_bf16kv.out")
ERR  = os.path.join(J, "dl", "vllm_bf16kv.err")
PORT = 8143
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--kv-cache-dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "2048", "--max-num-seqs", "4",
        "--enforce-eager", "--port", str(PORT), "--host", "127.0.0.1"]
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe)
print("server pid=%d port=%d (KV=bf16, chat, thinking off)" % (proc.pid, PORT), flush=True)

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

def post(path, payload, timeout=1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

mid = json.loads(urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % PORT, timeout=60).read())["data"][0]["id"]
payload = {"model": mid, "messages": [{"role": "user", "content": ZH}],
           "max_tokens": 96, "temperature": 0.0, "ignore_eos": True,
           "top_p": 1.0, "seed": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
resp = post("/v1/chat/completions", payload)
msg = resp["choices"][0].get("message", {})
content = msg.get("content") or ""
print("--- [bf16-kv] %.1fs tokens=%s/%s" % (time.time()-t0, resp["usage"]["prompt_tokens"],
                                            resp["usage"]["completion_tokens"]), flush=True)
print("    content[:150]=%r" % content[:150], flush=True)
with open(os.path.join(J, "dl", "vllm_bf16kv_text.txt"), "w", encoding="utf-8") as f:
    f.write(content)
print("VLLM_BF16KV_DONE", flush=True)
try: proc.kill()
except Exception: pass
