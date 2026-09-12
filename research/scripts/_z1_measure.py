#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z1: vLLM DSpark speculative-decoding acceptance measurement (Windows side).

Target  : models\\Qwen3.8-27B-NVFP4-RTX5090  (gittensor-model-hub, NVFP4/modelopt)
Draft   : data\\draft_model                   (Qwen3DSparkModel, 45 F32 tensors)
Recipe  : adapted 1:1 from the proven local launcher `launcher-dspark.py`
          (same env scrub / vcvars / cache dirs / quant flags), plus
          --port/--host, a small max-model-len and --enforce-eager so that
          nothing JITs (no nvcc / cl during the run).
Measures: vllm:spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total
          and vllm:spec_decode_num_accepted_tokens_per_pos_total{position=..}
Run with: C:\\vllm\\venv\\Scripts\\python.exe _z1_measure.py
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HERE = r"C:\Users\User\Documents\ziqinzhang"
TARGET = os.path.join(HERE, "models", "Qwen3.8-27B-NVFP4-RTX5090")
DRAFT = os.path.join(HERE, "data", "draft_model")
LOG = os.path.join(HERE, "_z1_server.log")
RESULT = os.path.join(HERE, "_z1_result.json")
PY = r"C:\vllm\venv\Scripts\python.exe"
SERVED = "qwen3.8-27b"
PORT = int(os.environ.get("Z1_PORT", "8100"))
BASE = "http://127.0.0.1:%d" % PORT

PROMPT = "\u8bf7\u7528\u4e2d\u6587\u5199\u4e00\u6bb5\u4e24\u767e\u5b57\u5de6\u53f3\u7684\u77ed\u6587\uff0c\u4ecb\u7ecd\u897f\u6e56\u4e00\u5e74\u56db\u5b63\u7684\u666f\u8272\u53d8\u5316\uff0c\u8981\u6c42\u8bed\u53e5\u8fde\u8d2f\u3001\u4e0d\u8981\u5217\u6761\u76ee\u3002"
MAXNEW = int(os.environ.get("Z1_MAXNEW", "96"))
MAXLEN = os.environ.get("Z1_MAXLEN", "4096")
GMU = os.environ.get("Z1_GMU", "0.90")
NUMSPEC = int(os.environ.get("Z1_K", "7"))
KVDT = os.environ.get("Z1_KVDT", "nvfp4")   # ninfer used --kv-dtype nvfp4
EAGER = os.environ.get("Z1_EAGER", "1") == "1"

SPEC_CONFIG = {
    "method": "dspark",
    "model": DRAFT.replace("\\", "/"),
    "num_speculative_tokens": NUMSPEC,
    "draft_sample_method": "greedy",
}


def build_env():
    env = dict(os.environ)
    # TEMP must be ASCII (sandbox denies non-ASCII TEMP on child launches)
    env["TEMP"] = r"C:\vllm\tmp"
    env["TMP"] = r"C:\vllm\tmp"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["VLLM_LOOPBACK_IP"] = "127.0.0.1"
    env["VLLM_DP_MASTER_IP"] = "127.0.0.1"
    env["FLASHINFER_CACHE_DIR"] = os.path.join(HERE, "data", "_fij_cache")
    env["VLLM_CACHE_ROOT"] = os.path.join(HERE, "data", ".vllm_cache")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(HERE, "data", ".inductor_cache")
    env["TRITON_CACHE_DIR"] = os.path.join(HERE, "data", ".triton_cache")
    env["NVFP4_MMA_DLL"] = os.path.join(r"C:\vllm\tmp", "nvfp4_mma_v15.dll")
    env["VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE"] = "0"
    env["VSLANG"] = "1033"
    for k in list(env):
        if any(s in k.upper() for s in ("MASTER_ADDR", "MASTER_PORT", "MASTER_IP")):
            if k.upper() not in ("VLLM_DP_MASTER_IP",):
                env.pop(k, None)
    try:
        sys.path.insert(0, HERE)
        from vcvars_env import get_vcvars_env
        env.update(get_vcvars_env())
    except Exception as e:
        print("vcvars_env failed (non-fatal):", e, flush=True)
    for k in ("CPATH", "LIBRARY_PATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
              "OBJC_INCLUDE_PATH"):
        env.pop(k, None)
    return env


def build_cmd():
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server", "--model", TARGET,
        "--served-model-name", SERVED,
        "--quantization", "modelopt",
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--max-model-len", MAXLEN,
        "--max-num-batched-tokens", MAXLEN,
        "--max-num-seqs", "4",
        "--gpu-memory-utilization", GMU,
        "--skip-mm-profiling",
        "--no-enable-flashinfer-autotune",
        "--master-addr", "127.0.0.1",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--speculative-config", json.dumps(SPEC_CONFIG),
    ]
    if KVDT:
        cmd += ["--kv-cache-dtype", KVDT]
    if EAGER:
        cmd += ["--enforce-eager"]
    return cmd


def metrics():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception:
        return None


def parse_spec(text):
    if not text:
        return None
    drafts = draft_tokens = accepted = 0
    per_pos = {}
    raw = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or not line.startswith("vllm:spec_decode"):
            continue
        raw.append(line)
        name = line.split("{")[0].split()[0]
        if not name.endswith("_total"):
            continue
        try:
            val = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if "num_accepted_tokens_per_pos" in name:
            m = re.search(r'position="(\d+)"', line)
            if m:
                per_pos[int(m.group(1))] = int(val)
        elif "num_draft_tokens" in name:
            draft_tokens += int(val)
        elif "num_accepted_tokens" in name:
            accepted += int(val)
        elif "num_drafts" in name:
            drafts += int(val)
    return drafts, draft_tokens, accepted, per_pos, raw


def wait_health(proc, timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print("!! server process exited rc=%s" % proc.returncode, flush=True)
            return False
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    env = build_env()
    cmd = build_cmd()
    print("CMD =", " ".join(cmd), flush=True)
    logf = open(LOG, "wb")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    print("server pid =", proc.pid, "-> log", LOG, flush=True)
    rc = 3
    try:
        if not wait_health(proc):
            return 2
        print("server healthy at", time.strftime("%H:%M:%S"), flush=True)

        b = parse_spec(metrics())
        print("BEFORE:", b[:4] if b else None, flush=True)

        body = json.dumps({
            "model": SERVED, "prompt": PROMPT, "max_tokens": MAXNEW,
            "temperature": 0.0, "top_p": 1.0, "n": 1, "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            BASE + "/v1/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read().decode("utf-8"))
        el = time.time() - t0
        a = parse_spec(metrics())
        choice = resp["choices"][0]
        text = choice.get("text", "")
        usage = resp.get("usage", {})
        print("wall = %.2f s  prompt_tokens=%s completion_tokens=%s"
              % (el, usage.get("prompt_tokens"), usage.get("completion_tokens")),
              flush=True)

        out = {
            "cmd": cmd, "spec_config": SPEC_CONFIG, "kv_cache_dtype": KVDT,
            "enforce_eager": EAGER, "gpu_memory_utilization": GMU,
            "max_model_len": int(MAXLEN), "prompt": PROMPT,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "wall_time_s": el, "generated_text": text,
            "before": ({"drafts": b[0], "draft_tokens": b[1], "accepted": b[2],
                        "per_pos": b[3]} if b else None),
            "after": ({"drafts": a[0], "draft_tokens": a[1], "accepted": a[2],
                       "per_pos": a[3]} if a else None),
        }
        if a and b:
            dd, dt, da = a[0] - b[0], a[1] - b[1], a[2] - b[2]
            hi = max(max(a[3]) if a[3] else 0, max(b[3]) if b[3] else 0) + 1
            dp = [a[3].get(i, 0) - b[3].get(i, 0) for i in range(hi)]
            out["delta"] = {
                "num_drafts": dd, "num_draft_tokens": dt, "num_accepted_tokens": da,
                "acceptance_rate_pct": (100.0 * da / dt) if dt else None,
                "acceptance_length": (1.0 + da / dd) if dd else None,
                "per_pos_counts": dp,
                "per_pos_rate_pct": [(100.0 * x / dd) if dd else None for x in dp],
            }
            print("\n================ RESULT ================")
            print("num_drafts        =", dd)
            print("num_draft_tokens  =", dt)
            print("num_accepted      =", da)
            print("acceptance_rate   = %.4f %%   (= accepted/draft_tokens)"
                  % out["delta"]["acceptance_rate_pct"])
            print("acceptance_length = %.4f      (= 1 + accepted/drafts)"
                  % out["delta"]["acceptance_length"])
            print("per_pos counts    =", dp)
            print("per_pos rate(%%)   =",
                  ["%.2f" % x if x is not None else None
                   for x in out["delta"]["per_pos_rate_pct"]])
            print("========================================", flush=True)
        out["raw_metrics_lines_after"] = a[4] if a else None
        with open(RESULT, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("wrote", RESULT, flush=True)
        print("----- generated text -----")
        print(text, flush=True)
        print("----- /metrics vllm:spec_decode* (after) -----")
        for l in (out["raw_metrics_lines_after"] or []):
            print(l, flush=True)
        rc = 0
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        logf.close()
        with open(LOG, "rb") as f:
            lines = f.read().decode("utf-8", "replace").splitlines()
        print("\n----- server log tail (last 60 lines) -----", flush=True)
        for l in lines[-60:]:
            print(l, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
