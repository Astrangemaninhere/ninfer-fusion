#!/usr/bin/env python3
"""正确解码下的同源判定：只比"两边 dtype 与形状都相同"的张量。
BF16 用 bf16 解码打印真实值；同时给出字节 md5 与逐值差异统计。
"""
import json, pathlib, struct, hashlib

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}

idx = json.loads((HF / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdr = {}
def hf_bytes(name):
    sh = wm.get(name)
    if not sh: return None, None
    if sh not in hdr:
        with open(HF / sh, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = (json.loads(f.read(hn).decode("utf-8")), hn)
    h, hn = hdr[sh]
    e = h.get(name)
    if not e: return None, None
    with open(HF / sh, "rb") as f:
        f.seek(8 + hn + e["data_offsets"][0])
        return f.read(e["data_offsets"][1] - e["data_offsets"][0]), e["dtype"]

def ninfer_bytes(name):
    o = objs[name]
    with open(NINFER, "rb") as f:
        f.seek(o["offset"]); return f.read(o["bytes"]), o["format"]

def bf16_vals(buf, k=6):
    out = []
    for i in range(0, min(len(buf), 2*k), 2):
        u = struct.unpack("<H", buf[i:i+2])[0]
        out.append(struct.unpack("<f", struct.pack("<I", u << 16))[0])
    return out

def f32_vals(buf, k=6):
    return list(struct.unpack("<%df" % min(k, len(buf)//4), buf[:4*min(k, len(buf)//4)]))

P = "model.language_model.layers.%d."
pairs = []
for L in (0, 1, 3, 5, 30, 63):
    for nin, suf in (("text/layers/%d/input_norm" % L, "input_layernorm.weight"),
                     ("text/layers/%d/post_attention_norm" % L, "post_attention_layernorm.weight"),
                     ("text/layers/%d/gdn/norm" % L, "linear_attn.norm.weight"),
                     ("text/layers/%d/attention/query_norm" % L, "self_attn.q_norm.weight")):
        if nin in objs:
            pairs.append((nin, (P % L) + suf))
pairs += [("text/final_norm", "model.language_model.norm.weight")]

same = diff = 0
for a, b in pairs:
    if a not in objs: continue
    ab, afmt = ninfer_bytes(a)
    bb, bdt = hf_bytes(b)
    if bb is None:
        print("  [HF 无] %s" % b); continue
    tag = "%s  (%s vs %s)" % (a.split("/", 1)[1], afmt, bdt)
    if len(ab) != len(bb):
        print("  [长度不同] %-58s %d vs %d" % (tag, len(ab), len(bb))); diff += 1; continue
    if ab == bb:
        print("  [完全一致] %-58s 前值=%s" % (tag, bf16_vals(ab, 3))); same += 1
    else:
        diff += 1
        va, vb = bf16_vals(ab, 4), bf16_vals(bb, 4)
        md5a, md5b = hashlib.md5(ab).hexdigest()[:10], hashlib.md5(bb).hexdigest()[:10]
        print("  [不同]     %-58s" % tag)
        print("        ninfer md5=%s 前4=%s" % (md5a, [round(x, 6) for x in va]))
        print("        hf     md5=%s 前4=%s" % (md5b, [round(x, 6) for x in vb]))

print("\n=== 判定 ===")
print("  完全一致 %d 项，不同 %d 项" % (same, diff))
if same and not diff:
    print("  => 同源：下载的 HF 检查点就是 ninfer 产物里的那份目标权重。")
elif same and diff:
    print("  => 混合：部分一致、部分不同 —— 需要逐个看不同的那些（可能是 layout/转置而非权重差异）。")
else:
    print("  => 全部不同：血统不同（或我的名映射/偏移仍不对）。")
