import json, struct, pathlib

MODELS = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\models")
for d in ["Qwen3.8-27B-NVFP4-RTX5090", "Qwen3.8-27B-NVFP4", "Qwen3.8-Flash-Next-ABLITERATED-NVFP4"]:
    p = MODELS / d
    if not p.exists():
        print("== %s : 目录不存在" % d); continue
    print("== %s" % d)
    # 列出 index / 单个 safetensors
    idx = p / "model.safetensors.index.json"
    shards = []
    if idx.exists():
        j = json.loads(idx.read_text(encoding="utf-8", errors="replace"))
        wm = j.get("weight_map", {})
        hits = {k: v for k, v in wm.items() if "draft" in k or "mtp" in k.lower()}
        print("   index 张量数 =", len(wm), " 含 draft/mtp =", len(hits))
        for k, v in list(hits.items())[:12]:
            print("     %-60s -> %s" % (k, v))
        shards = sorted(set(wm.values()))
    else:
        shards = [f.name for f in sorted(p.glob("*.safetensors"))]
    print("   shards =", shards[:4])
    # 直接读每个 shard 的头，找 draft_head 及其 dtype
    for s in shards:
        f = p / s
        if not f.exists():
            continue
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if "draft_head" in k or ("draft" in k and "head" in k):
                sz = v["data_offsets"][1] - v["data_offsets"][0]
                print("   ★ %-52s dtype=%-8s shape=%-20s %d B  (%s)" %
                      (k, v.get("dtype"), str(v.get("shape")), sz, s))
