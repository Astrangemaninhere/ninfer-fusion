#!/usr/bin/env python3
"""参考跑：vllm 0.29（Linux/WSL）+ dflash2 草稿（incoai/Qwen3.8-27B-DFlash2）+ NVFP4 target。
贪心请求，抓 /metrics 的 spec_decode 计数与逐位置剖面（与 ninfer 探针同口径）。"""
import json, os, re, subprocess, sys, time, urllib.request

PY     = "/home/user/vllm029/bin/python"
J      = "/mnt/c/Users/User/Documents/ziqinzhang"
TARGET = "/home/user/models/qwen3_8_27b_nvfp4_hf"
DRAFT  = "/home/user/models/draft_dflash2_ref"
OUT    = os.path.join(J, "dl", "vref_df2c.out")
ERR    = os.path.join(J, "dl", "vref_df2c.err")
PORT   = 8130
NTOK   = 7
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"

env = dict(os.environ)
env["HF_HUB_OFFLINE"] = "1"
env["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"   # WSL 默认 False -> UVA 不可用 (platforms/interface.py:1002)
env["VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY"] = "0"          # HF 不可达（已实测），避免联网等待
env.setdefault("VLLM_LOGGING_LEVEL", "INFO")

spec = {"method": "dflash", "model": DRAFT, "num_speculative_tokens": NTOK,
        "draft_sample_method": "greedy"}
args = [PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", TARGET,
        "--speculative-config", json.dumps(spec),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--max-num-seqs", "16",
        "--enforce-eager",
        "--port", str(PORT), "--host", "127.0.0.1"]
print("spec = %s" % json.dumps(spec), flush=True)
print("target exists=%s draft exists=%s" % (os.path.exists(TARGET), os.path.exists(DRAFT)), flush=True)
fo = open(OUT, "w", encoding="utf-8", errors="replace")
fe = open(ERR, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(args, stdout=fo, stderr=fe, env=env, cwd=J)
print("server pid=%d port=%d" % (proc.pid, PORT), flush=True)

def tail(path, n=10):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return ""

ready = False
for i in range(240):                  # 最多 1200s（模型加载 + 编译/调优）
    time.sleep(5)
    t = tail(ERR, 12) + tail(OUT, 10)
    if re.search(r"Application startup complete|Uvicorn running on|Started server process", t):
        ready = True; break
    if re.search(r"Traceback \(most recent call last\)|Error|error:", t) and i > 3:
        # 只在明显致命时提前退出（"Error" 太宽，故要求同时出现 Traceback 或 engine 崩溃）
        if "Traceback" in t or "EngineCore" in t and "died" in t:
            print("startup error at %ds" % (i*5), flush=True); break
    if proc.poll() is not None:
        print("server exited rc=%s at %ds" % (proc.returncode, i*5), flush=True); break
    if i % 12 == 11:
        print("  ... %ds last: %s" % ((i+1)*5, tail(ERR, 1).strip()[:120]), flush=True)
print("ready=%s at %s" % (ready, time.strftime("%H:%M:%S")), flush=True)
if not ready:
    print("=== err tail 30 ===", flush=True); print(tail(ERR, 30), flush=True)
    try: proc.kill()
    except Exception: pass
    sys.exit(2)

def metrics():
    with urllib.request.urlopen("http://127.0.0.1:%d/metrics" % PORT, timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    acc = drf = 0.0; per = {}
    for l in [x.strip() for x in txt.splitlines() if "spec_decode" in x and not x.startswith("#")]:
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
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.load(r)
    dt = time.time() - t0
    ch = resp["choices"][0]
    a, d, per = metrics()
    print("--- [%s] %.1fs finish=%s prompt=%s completion=%s" %
          (tag, dt, ch.get("finish_reason"), resp["usage"]["prompt_tokens"],
           resp["usage"]["completion_tokens"]), flush=True)
    print("    text[:60]=%r" % ch["text"][:60], flush=True)
    if d > 0:
        print("    ACCEPTED=%d DRAFTED=%d RATE=%.2f%% PER_POS=%s" %
              (a, d, 100.0*a/d, [per.get(i, 0) for i in range(NTOK)]), flush=True)

try:
    ask("hello", "hello")
    ask(ZH, "zh")
except Exception as e:
    print("request error: %s" % e, flush=True)

print("=== 日志里的 SpecDecoding 行 ===", flush=True)
for l in tail(OUT, 60).splitlines():
    if "SpecDecoding" in l:
        print("  " + l.strip()[:220], flush=True)
print("VREF_DF2_DONE", flush=True)
try: proc.kill()
except Exception: pass
