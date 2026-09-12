#!/usr/bin/env python3
"""直接打活着的服务（port 8130）：贪心请求 + 抓 spec_decode 逐位置指标（不重载模型）"""
import json, re, sys, time, urllib.request

PORT = 8130
ZH = "请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。"
BASE = "http://127.0.0.1:%d" % PORT

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
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

print("=== 0) 服务健康检查 ===")
try:
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        print("  models:", r.read().decode()[:200])
except Exception as e:
    print("  服务不可用:", type(e).__name__, e); sys.exit(2)

a0, d0, p0 = metrics()
print("=== 1) 请求前指标: accepted=%s drafted=%s per_pos=%s" % (a0, d0, p0))

def ask(prompt, tag):
    body = json.dumps({"model": "qwen3", "prompt": prompt, "max_tokens": 96,
                       "temperature": 0.0, "ignore_eos": True}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.load(r)
    dt = time.time() - t0
    ch = resp["choices"][0]
    a, d, per = metrics()
    print("--- [%s] %.1fs finish=%s prompt=%s completion=%s" %
          (tag, dt, ch.get("finish_reason"), resp["usage"]["prompt_tokens"],
           resp["usage"]["completion_tokens"]))
    print("    text[:70]=%r" % ch["text"][:70])
    da, dd = a - a0, d - d0
    if dd > 0:
        print("    [增量] ACCEPTED=%d DRAFTED=%d  ACCEPTANCE=%.2f%%" % (da, dd, 100.0*da/dd))
        print("    [增量] PER_POS=%s" % [(k, round(per.get(k, 0) - p0.get(k, 0), 1)) for k in sorted(per)][:8])
    else:
        print("    [累计] ACCEPTED=%d DRAFTED=%d  ACCEPTANCE=%.2f%% PER_POS=%s" %
              (a, d, 100.0*a/d if d else 0, [per.get(k, 0) for k in sorted(per)][:8]))

try:
    ask(ZH, "zh")
except Exception as e:
    print("请求失败:", type(e).__name__, str(e)[:300])
