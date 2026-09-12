#!/usr/bin/env python3
"""判定：破禁版 .ninfer 的目标权重 vs 下载的非破禁 Qwen3.8-27B-NVFP4-RTX5090。
若两者不同 ⇒ "草稿对着原版训、部署喂破禁版" 就是接受率崩的根因。
用与之前相同的方法（BF16/FP32 张量逐字节比，含 payload_start = align_up(16+json_len,4096)）。"""
import json, pathlib, struct, hashlib

HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")
CAND = [
    ("abliterated_nvfp4", "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-NVFP4/qwen3_8_27b_nvfp4.ninfer"),
    ("local_nvfp4",       "/home/user/models/qwen3_8_27b_nvfp4.ninfer"),
]

def align_up(x, a): return ((x + a - 1) // a) * a

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

P = "model.language_model.layers.%d."
pairs = [("text/final_norm", "model.language_model.norm.weight")]
for L in (0, 3, 5, 30, 63):
    pairs += [("text/layers/%d/input_norm" % L, (P % L) + "input_layernorm.weight"),
              ("text/layers/%d/post_attention_norm" % L, (P % L) + "post_attention_layernorm.weight"),
              ("text/layers/%d/gdn/norm" % L, (P % L) + "linear_attn.norm.weight")]

for tag, path in CAND:
    p = pathlib.Path(path)
    print("=== %s ===" % tag)
    if not p.exists():
        print("  缺: %s" % path); print(); continue
    with open(p, "rb") as f:
        f.read(8); n = struct.unpack("<Q", f.read(8))[0]
        man = json.loads(f.read(n).decode("utf-8", "replace"))
    payload = align_up(16 + n, 4096)
    objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}
    print("  identity=%s  payload_start=%d" % (json.dumps(man.get("identity"), ensure_ascii=False), payload))
    same = diff = miss = 0
    for a, b in pairs:
        if a not in objs:
            miss += 1; continue
        o = objs[a]
        with open(p, "rb") as f:
            f.seek(payload + o["offset"]); A = f.read(o["bytes"])
        B = hfb(b)
        if B is None or len(A) != len(B):
            miss += 1; continue
        if A == B:
            same += 1
        else:
            diff += 1
            if diff <= 3:
                print("   [不同] %-42s ninfer md5=%s hf md5=%s" %
                      (a.split("/", 1)[1], hashlib.md5(A).hexdigest()[:10], hashlib.md5(B).hexdigest()[:10]))
    print("  ==> 一致 %d / 不同 %d / 缺 %d" % (same, diff, miss))
    if diff == 0 and same > 0:
        print("      与下载的原版【完全一致】⇒ 它就是原版，草稿与目标同源")
    elif diff > 0:
        print("      【与原版不同】⇒ 若草稿是对原版训的，这就是接受率崩的根因")
    print()
