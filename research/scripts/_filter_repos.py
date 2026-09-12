import json, os, urllib.request, urllib.parse

BASE = "https://hf-mirror.com"
UA = {"User-Agent": "p"}

def jget(path, timeout=25):
    req = urllib.request.Request(BASE + path, headers=UA)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

cands = [
    "Inferact/Qwen3.8-27B-NVFP4",
    "RadixArk/Qwen3.8-27B-NVFP4",
    "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead",
    "unsloth/Qwen3.8-27B-NVFP4",
    "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
    "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
    "sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4",
]
# 再搜一轮：破禁 + nvfp4
for q in ["Qwen3.8-27B abliterated nvfp4", "Qwen3.8-27B uncensored nvfp4", "Qwen3.8-27B heretic NVFP4"]:
    try:
        for m in jget("/api/models?search=" + urllib.parse.quote(q) + "&limit=30"):
            if "nvfp4" in m["id"].lower() and m["id"] not in cands:
                cands.append(m["id"])
    except Exception as e:
        print("search fail", q, e)

print("=== 逐仓核验：是否 双组(FP8+NVFP4)、lm_head/mtp 是否 bf16、体积 ===")
print(f"{'repo':<52} {'组':<16} {'lm_head':<8} {'mtp':<6} {'GB':<7} 破禁")
for mid in cands:
    try:
        c = jget(f"/{mid}/raw/main/config.json")
    except Exception as e:
        print(f"{mid:<52} config 取不到 ({type(e).__name__})")
        continue
    qc = c.get("quantization_config") or {}
    cg = qc.get("config_groups") or {}
    desc = []
    for g, v in cg.items():
        v = v or {}
        w = v.get("weights") or {}
        nb = w.get("num_bits")
        desc.append(f"{g}:{nb}b")
    ign = [str(x) for x in (qc.get("ignore") or [])]
    lmb = any("lm_head" in x for x in ign)
    mtpb = any("mtp" in x.lower() for x in ign)
    # 体积
    try:
        d = jget(f"/api/models/{mid}?blobs=true")
        sz = sum((s.get("size") or 0) for s in (d.get("siblings") or [])) / 1e9
    except Exception:
        sz = float("nan")
    low = mid.lower()
    ab = "abli" in low or "uncens" in low or "heretic" in low or "huihui" in low
    print(f"{mid:<52} {','.join(desc):<16} {str(lmb):<8} {str(mtpb):<6} {sz:<7.1f} {'YES' if ab else '-'}")
    # 越详细越好：打印每组的 targets 与 input_activations
    for g, v in cg.items():
        v = v or {}
        w = v.get("weights") or {}
        ia = v.get("input_activations") or {}
        print(f"      {g}: w.format={w.get('format') or v.get('format')} bits={w.get('num_bits')} "
              f"grp={w.get('group_size')} ia.bits={ia.get('num_bits')} ia.type={ia.get('type')} "
              f"targets={str(v.get('targets'))[:70]}")
