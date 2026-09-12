import json, urllib.request

BASE = "https://hf-mirror.com"
UA = {"User-Agent": "p"}

def jget(path, timeout=25):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE + path, headers=UA), timeout=timeout).read().decode())

cands = [
    "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
    "huihui-ai/Qwen3.8-27B-abliterated",
    "orcarouter/Qwen3.8-27B-Uncensored",
    "Qwen/Qwen3.8-27B",
]
print("=== bf16 破禁源候选核验（无 quantization_config 才可作为 --model）===")
for mid in cands:
    try:
        d = jget(f"/api/models/{mid}?blobs=true")
    except Exception as e:
        print(f"  {mid:<48} 不可得 ({type(e).__name__})")
        continue
    sib = d.get("siblings") or []
    tot = sum((s.get("size") or 0) for s in sib) / 1e9
    shard = sum(1 for s in sib if s["rfilename"].endswith(".safetensors"))
    line = f"  {mid:<48} {tot:>6.1f} GB  分片{shard:<3} gated={d.get('gated')}"
    try:
        c = jget(f"/{mid}/raw/main/config.json")
        qc = c.get("quantization_config")
        tc = c.get("text_config") or {}
        line += (f"  quant={qc is not None}  dtype={tc.get('dtype') or c.get('dtype')}"
                 f"  arch={c.get('architectures')}")
        # 是否自带 mtp / draft_head
        names = " ".join(s["rfilename"] for s in sib).lower()
        line += f"  mtp文件={'mtp' in names}"
    except Exception as e:
        line += f"  config 读取失败 {type(e).__name__}"
    print(line)
