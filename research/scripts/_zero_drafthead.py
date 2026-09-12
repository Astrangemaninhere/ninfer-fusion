import json, struct, pathlib, sys

# 原地清零原生盘 refhead 副本里的 text/draft_head 码（保留 header 与其它对象）
ART = pathlib.Path("/home/user/models/qwen3_8_27b_nvfp4_dflash2_refhead.ninfer")
with open(ART, "rb") as f:
    magic = f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
o = objs.get("text/draft_head")
print("magic =", magic, " json_len =", jlen, " payload_start =", payload)
if o is None:
    print("text/draft_head 不存在"); sys.exit(2)
print("text/draft_head:", {k: o.get(k) for k in ("kind", "format", "shape", "bytes")})

nbytes = o["bytes"]
shape = o.get("shape") or []
n_elem = (shape[0] * shape[1]) if len(shape) == 2 else None
print("  元素数 =", n_elem, " 4bit码字节 = %s  余下(scale) = %s" %
      (n_elem // 2 if n_elem else "?", nbytes - (n_elem // 2) if n_elem else "?"))

# 只清 4bit 码部分，保留 scale（若布局是 码在前 / scale 在后）
with open(ART, "r+b") as f:
    f.seek(payload + o["offset"])
    f.write(b"\x00" * (n_elem // 2))
print("已清零前 %d 字节（4bit 码区）" % (n_elem // 2))

# 回读确认
with open(ART, "rb") as f:
    f.seek(payload + o["offset"])
    head = f.read(1 << 16)
    f.seek(payload + o["offset"] + n_elem // 2 - 16)
    tail = f.read(32)
print("  码区前 64B 全零?", all(b == 0 for b in head[:64]))
print("  码区末尾 16B 全零?", all(b == 0 for b in tail[:16]))
print("ZERO_DRAFTHEAD_DONE")
