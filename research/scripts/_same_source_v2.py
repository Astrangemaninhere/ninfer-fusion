#!/usr/bin/env python3
"""同源判定（offset 已定标：payload_start = align_up(16+json_len, 4096)）。"""
import json, pathlib, struct

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
def align_up(x, a): return ((x + a - 1) // a) * a
PAYLOAD = align_up(16 + n, 4096)
print("hdr_end=%d  payload_start=%d" % (16 + n, PAYLOAD))
objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}

# 先验证：tokenizer.json 处应是 {"version"
with open(NINFER, "rb") as f:
    f.seek(PAYLOAD + 0); head = f.read(40)
print("验证 payload_start+0 = %r" % head)
assert head.lstrip()[:2] == b'{"', "载荷起点定标失败"
print("  定标 OK\n")

idx = json.loads((HF / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]; hdr = {}
def hfb(name):
    sh = wm.get(name)
    if not sh: return None
    if sh not in hdr:
        with open(HF / sh, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = (json.loads(f.read(hn).decode("utf-8")), hn)
    h, hn = hdr[sh]; e = h.get(name)
    if not e: return None
    with open(HF / sh, "rb") as f:
        f.seek(8 + hn + e["data_offsets"][0])
        return f.read(e["data_offsets"][1] - e["data_offsets"][0])

def nfb(name):
    o = objs[name]
    with open(NINFER, "rb") as f:
        f.seek(PAYLOAD + o["offset"]); return f.read(o["bytes"])

def bvals(b, k=4):
    out = []
    for i in range(0, min(len(b), 2*k), 2):
        u = struct.unpack("<H", b[i:i+2])[0]
        out.append(round(struct.unpack("<f", struct.pack("<I", u << 16))[0], 6))
    return out

P = "model.language_model.layers.%d."
pairs = [("text/final_norm", "model.language_model.norm.weight")]
for L in (0, 1, 3, 5, 30, 63):
    for a, s in (("text/layers/%d/input_norm" % L, "input_layernorm.weight"),
                 ("text/layers/%d/post_attention_norm" % L, "post_attention_layernorm.weight"),
                 ("text/layers/%d/gdn/norm" % L, "linear_attn.norm.weight"),
                 ("text/layers/%d/attention/query_norm" % L, "self_attn.q_norm.weight")):
        if a in objs: pairs.append((a, (P % L) + s))

same = diff = 0
for a, b in pairs:
    if a not in objs: continue
    A = nfb(a); B = hfb(b)
    if B is None: print("  [HF无] %s" % b); continue
    if len(A) != len(B):
        print("  [长度异] %-44s %d vs %d" % (a.split("/", 1)[1], len(A), len(B))); diff += 1; continue
    if A == B:
        same += 1
        print("  [一致]   %-44s %s" % (a.split("/", 1)[1], bvals(A, 3)))
    else:
        diff += 1
        print("  [不同]   %-44s" % a.split("/", 1)[1])
        print("        ninfer %s" % bvals(A, 4))
        print("        hf     %s" % bvals(B, 4))

print("\n=== 判定: 一致 %d, 不同 %d ===" % (same, diff))
print("  => 同源，可直接用下载件做参照" if same and not diff else
      ("  => 混合" if same else "  => 全不同：血统不同，必须反拆"))
