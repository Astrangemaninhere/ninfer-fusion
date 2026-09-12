import json, pathlib, struct

D = pathlib.Path("/home/user/models/q38_abl_huihui_nvfp4")
idx = D / "model.safetensors.index.json"
print("=== 已下载文件 ===")
for f in sorted(D.iterdir()):
    if f.is_file():
        print(f"  {f.name:<44} {f.stat().st_size/1e6:>9.1f} MB")
print()
if not idx.exists():
    print("index.json 尚未到（下载中）")
    raise SystemExit(0)

wm = json.loads(idx.read_text()).get("weight_map") or {}
print(f"=== index 里共 {len(wm)} 个张量键 ===")
groups = {
    "text/token_embedding": ("embed_tokens",),
    "text/draft_head": ("draft_head", "draft_model", "selector", "codebook"),
    "mtp/": ("mtp",),
    "vision/": ("visual", "vision", "patch_embed", "pos_embed"),
    "lm_head": ("lm_head",),
}
for label, pats in groups.items():
    hits = sorted(k for k in wm if any(p in k.lower() for p in pats))
    print(f"--- {label}: {len(hits)} 个")
    for k in hits[:10]:
        print(f"    {k:<66} -> {wm[k]}")

def headers(fn):
    p = D / fn
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n).decode("utf-8"))

print()
print("=== 关键张量的真实 dtype（只读已下载分片的 header）===")
want = ("lm_head", "embed_tokens", "mtp")
for k in sorted(wm):
    if not any(w in k.lower() for w in want):
        continue
    fn = wm[k]
    p = D / fn
    if not p.exists():
        print(f"  {k:<58} ({fn} 未下载完)")
        continue
    try:
        h = headers(fn)
        m = h.get(k) or {}
        print(f"  {k:<58} dtype={m.get('dtype'):<9} shape={tuple(m.get('shape') or ())}")
    except Exception as e:
        print(f"  {k:<58} header 读取失败 {e}")
