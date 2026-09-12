import json, pathlib, struct

D = pathlib.Path("/home/user/models/q3nvfp4")
idx = json.loads((D / "model.safetensors.index.json").read_text())
wm = idx.get("weight_map") or {}
print("=== 源 safetensors 里与 head/embed 相关的键 ===")
keys = [k for k in wm if any(s in k for s in ("lm_head", "embed_tokens", "mtp", "draft"))]
for k in sorted(keys)[:40]:
    print(f"  {k:<70} -> {wm[k]}")
print(f"  （共 {len(keys)} 个）")

# 读取相关分片的 header 元数据（safetensors: u64 len + JSON）
def headers(fn):
    p = D / fn
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n).decode("utf-8"))

files = sorted({wm[k] for k in keys} | set(wm.values()))
seen = {}
for fn in files:
    if not fn.endswith(".safetensors"):
        continue
    try:
        h = headers(fn)
    except Exception as e:
        print(f"  读 {fn} 头失败: {e}"); continue
    for name, meta in h.items():
        if name == "__metadata__":
            continue
        seen[name] = (meta.get("dtype"), tuple(meta.get("shape") or ()), fn)

print()
print("=== 每个 head/embed 键的全部相关张量（含 scale 伴随张量）===")
for prefix in ("lm_head", "model.language_model.embed_tokens", "mtp"):
    rel = sorted(n for n in seen if n.startswith(prefix))
    print(f"--- {prefix} ---")
    for n in rel[:14]:
        dt, sh, fn = seen[n]
        print(f"    {n:<64} dtype={dt:<8} shape={sh}")
print()
print("=== 全局 dtype 直方图（源）===")
from collections import Counter
c = Counter(v[0] for v in seen.values())
for k, v in c.most_common():
    print(f"  {str(k):<10} {v}")
print()
print("=== 是否有 bf16 的 lm_head（关键判定）===")
lm = {n: seen[n] for n in seen if "lm_head" in n}
if not lm:
    print("  源里没有任何 lm_head 张量（可能命名为 output_head / head）")
for n, (dt, sh, fn) in lm.items():
    print(f"  {n}: dtype={dt} shape={sh} file={fn}")
