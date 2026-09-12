#!/usr/bin/env python3
"""涓?ninfer 鍚?prompt 鍚屽弬鏁扮殑 vLLM 瀵圭収娴嬮噺銆?ninfer 渚х敤鐨勬槸涓枃瑗挎箹 prompt / greedy / 96 token銆?""
import json, re, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8126
TARGET = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-NVFP4-RTX5090"
P = ("璇风敤涓枃鍐欎竴娈典袱鐧惧瓧宸﹀彸鐨勭煭鏂囷紝浠嬬粛瑗挎箹涓€骞村洓瀛ｇ殑鏅壊鍙樺寲锛?
     "瑕佹眰璇彞杩炶疮銆佷笉瑕佸垪鏉＄洰銆?)
BASE = "http://127.0.0.1:%d" % PORT

def snapshot():
    with urllib.request.urlopen(BASE + "/metrics", timeout=60) as r:
        m = r.read().decode("utf-8", "replace")
    acc = drf = dr = 0.0
    per = {}
    for l in m.splitlines():
        if l.startswith("#"):
            continue
        mm = re.search(r'spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\}\s+([0-9.eE+-]+)', l)
        if mm:
            per[int(mm.group(1))] = float(mm.group(2)); continue
        mm = re.search(r"spec_decode_num_accepted_tokens_total[^{]*\s+([0-9.eE+-]+)", l)
        if mm:
            acc = float(mm.group(1)); continue
        mm = re.search(r"spec_decode_num_draft_tokens_total[^{]*\s+([0-9.eE+-]+)", l)
        if mm:
            drf = float(mm.group(1)); continue
        mm = re.search(r"spec_decode_num_drafts_total[^{]*\s+([0-9.eE+-]+)", l)
        if mm:
            dr = float(mm.group(1))
    return acc, drf, dr, per

a0, d0, r0, p0 = snapshot()
print("=== 璇锋眰锛堜笌 ninfer 鍚?prompt / greedy / 96 token锛?==", flush=True)
body = json.dumps({"model": TARGET, "prompt": P, "max_tokens": 96,
                   "temperature": 0.0, "ignore_eos": True}).encode()
req = urllib.request.Request(BASE + "/v1/completions", data=body,
                            headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1200) as r:
    resp = json.load(r)
ch = resp["choices"][0]
print("  finish=%s prompt=%s completion=%s" % (ch.get("finish_reason"),
      resp["usage"]["prompt_tokens"], resp["usage"]["completion_tokens"]), flush=True)
print("  text[:60]=%r" % ch["text"][:60], flush=True)

a1, d1, r1, p1 = snapshot()
da, dd, dr = a1 - a0, d1 - d0, r1 - r0
dper = {k: p1.get(k, 0) - p0.get(k, 0) for k in set(list(p0) + list(p1))}
print()
print("=== 鏈澧為噺 ===", flush=True)
print("  drafts(杞? = %s   draft_tokens = %s   accepted = %s" % (dr, dd, da), flush=True)
if dd > 0:
    print("  === vLLM ACCEPTANCE = %.2f%% ===" % (100.0 * da / dd), flush=True)
print("  PER_POSITION(counts) = %s" % [int(dper.get(i, 0)) for i in range(9)], flush=True)
print("  锛坣infer 瀵圭収锛歞flash2 K=7 = 4.97%%, 鍓栭潰 [8,0,0,0,0,0,0]锛?
      "dspark K=7 = 7.63%%, 鍓栭潰 [17,1,0,0,0,0,0]锛?, flush=True)
print("CLIENT2_DONE", flush=True)
