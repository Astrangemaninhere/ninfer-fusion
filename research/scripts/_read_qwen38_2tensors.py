import json, struct, pathlib

p = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\Qwen3.8-27B\model-00001-of-00001.safetensors")
print("file:", p, "%.2f GiB" % (p.stat().st_size / 2**30))
print("index.json:", (p.parent / "model.safetensors.index.json").read_text(encoding="utf-8", errors="replace")[:200])
with open(p, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
for k, v in hdr.items():
    if k == "__metadata__":
        print("__metadata__ =", json.dumps(v, ensure_ascii=False)[:300]); continue
    d = v["data_offsets"][1] - v["data_offsets"][0]
    print("  %-40s dtype=%-10s shape=%-20s %d B" % (k, v.get("dtype"), str(v.get("shape")), d))
