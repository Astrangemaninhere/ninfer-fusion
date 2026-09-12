import json, os, urllib.request, urllib.parse

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
BASE = os.environ["HF_ENDPOINT"]

def get(path, timeout=25):
    url = BASE + path
    req = urllib.request.Request(url, headers={"User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

print("=== A) 镜像 API 搜索：真实存在的候选仓库 ===")
queries = ["Qwen3.8-27B-NVFP4", "Qwen3.8-27B abliterated", "Qwen3.8-27B uncensored"]
seen = {}
for q in queries:
    try:
        res = get("/api/models?search=" + urllib.parse.quote(q) + "&limit=40")
    except Exception as e:
        print(f"  查询 {q!r} 失败: {e}")
        continue
    for m in res:
        seen.setdefault(m["id"], m)
print(f"  共 {len(seen)} 个唯一仓库")
rows = []
for mid, m in seen.items():
    low = mid.lower()
    tag = []
    if "nvfp4" in low: tag.append("nvfp4")
    if any(k in low for k in ("abliterat", "uncensor", "huihui", "heretic", "aggressive")): tag.append("破禁")
    if "27b" in low: tag.append("27b")
    rows.append((mid, m.get("downloads", 0), m.get("likes", 0), ",".join(tag)))
rows.sort(key=lambda r: -r[1])
for mid, d, l, t in rows[:30]:
    print(f"  {mid:<64} dl={d:<8} like={l:<5} [{t}]")

print()
print("=== B) 逐个核验：lm_head 是否 bf16（在 quant ignore 名单里）、是否含 MTP 头 ===")
cands = [mid for mid, d, l, t in rows if "nvfp4" in mid.lower() and "27b" in mid.lower()]
cands = cands[:12] if cands else []
for mid in cands:
    print(f"  --- {mid} ---")
    try:
        info = get(f"/api/models/{mid}?blobs=true")
    except Exception as e:
        print(f"      取信息失败: {e}")
        continue
    sib = info.get("siblings") or []
    tot = sum((s.get("size") or 0) for s in sib)
    print(f"      文件 {len(sib)} 个，总 {tot/1e9:.1f} GB，gated={info.get('gated')}")
    try:
        cfg_raw = urllib.request.urlopen(
            urllib.request.Request(BASE + f"/{mid}/raw/main/config.json",
                                   headers={"User-Agent": "probe"}), timeout=25).read()
        cfg = json.loads(cfg_raw.decode("utf-8"))
        qc = cfg.get("quantization_config") or {}
        ign = qc.get("ignore") or qc.get("ignore_modules") or []
        ign = [str(x) for x in ign]
        lm = [x for x in ign if "lm_head" in x]
        mtp = [x for x in ign if "mtp" in x.lower()]
        print(f"      quant_method={qc.get('quant_method')} fmt={qc.get('format')} ignore 条目={len(ign)}")
        print(f"      lm_head 在 ignore = {bool(lm)}  例: {lm[:2]}")
        print(f"      mtp 在 ignore    = {bool(mtp)} 例: {mtp[:2]}")
        print(f"      architectures={cfg.get('architectures')} model_type={cfg.get('model_type')}")
    except Exception as e:
        print(f"      读 config 失败: {e}")
print("PROBE_DONE")
