#!/usr/bin/env python3
"""同源判定：把 ninfer 产物的 BF16/FP32 张量与下载的 HF 检查点逐字节比。
这些张量两边都是未量化格式，可直接比字节 —— 不需要任何反量化。
"""
import json, pathlib, struct, hashlib

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")
P = "model.language_model.layers.%d."

# ninfer 名 -> HF 名（只列可直接比字节的：两边同为 BF16/FP32 且形状相同）
def map_names(ninfer_name, hf_set):
    p = ninfer_name.split("/")
    if len(p) >= 3 and p[0] == "text" and p[1] == "layers":
        L, rest = int(p[2]), "/".join(p[3:])
        cands = {
            "input_norm":            P % L + "input_layernorm.weight",
            "post_attention_norm":   P % L + "post_attention_layernorm.weight",
            "gdn/norm":              P % L + "linear_attn.norm.weight",
            "gdn/a_log":             P % L + "linear_attn.A_log",
            "gdn/dt_bias":           P % L + "linear_attn.dt_bias",
            "gdn/convolution":       P % L + "linear_attn.conv1d.weight",
            "attention/query_norm":  P % L + "self_attn.q_norm.weight",
            "attention/key_norm":    P % L + "self_attn.k_norm.weight",
        }
        c = cands.get(rest)
        if c and c in hf_set:
            return c
    if ninfer_name == "text/final_norm":
        for c in ("model.language_model.norm.weight", "model.norm.weight"):
            if c in hf_set:
                return c
    if ninfer_name == "mtp/final_norm":
        return None  # mtp 侧在 HF 里未必对应
    return None

# --- 读 ninfer 清单 ---
with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}
plain = {k: v for k, v in objs.items()
         if v.get("format") in ("BF16", "FP32") and v.get("layout") == "contiguous-le-v1"}

# --- HF 定位 ---
idx = json.loads((HF / "model.safetensors.index.json").read_text(encoding="utf-8"))
wm = idx["weight_map"]
hdr = {}
def hf_read(name):
    sh = wm.get(name)
    if not sh: return None
    if sh not in hdr:
        with open(HF / sh, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr[sh] = (json.loads(f.read(hn).decode("utf-8")), hn)
    h, hn = hdr[sh]
    e = h.get(name)
    if not e: return None
    with open(HF / sh, "rb") as f:
        f.seek(8 + hn + e["data_offsets"][0])
        return f.read(e["data_offsets"][1] - e["data_offsets"][0])

def ninfer_read(name):
    o = objs[name]
    with open(NINFER, "rb") as f:
        f.seek(o["offset"]); return f.read(o["bytes"])

print("ninfer BF16/FP32 未量化张量: %d 个" % len(plain))
same = diff = 0
shown = 0
for nm in sorted(plain):
    hf_name = map_names(nm, wm)
    if not hf_name: continue
    a = ninfer_read(nm); b = hf_read(hf_name)
    if b is None: continue
    if len(a) != len(b):
        print("  [长度不同] %-42s %d vs %s %d" % (nm, len(a), hf_name, len(b)))
        diff += 1; continue
    if a == b:
        same += 1
        if shown < 6:
            print("  [逐字节一致] %-40s == %s" % (nm, hf_name.split("layers.")[-1])); shown += 1
    else:
        diff += 1
        if diff <= 8:
            da = struct.unpack("<" + "f"*(min(4, len(a)//4)), a[:16]) if True else ()
            db = struct.unpack("<" + "f"*(min(4, len(b)//4)), b[:16]) if True else ()
            print("  [不同] %-42s vs %s" % (nm, hf_name))
            print("        ninfer md5=%s 前4=%s" % (hashlib.md5(a).hexdigest()[:10],
                  [round(x,6) for x in da if abs(x) < 1e30][:4]))
            print("        hf     md5=%s 前4=%s" % (hashlib.md5(b).hexdigest()[:10],
                  [round(x,6) for x in db if abs(x) < 1e30][:4]))

print("\n=== 判定 ===")
print("  逐字节一致: %d    不同: %d" % (same, diff))
if same > 0 and diff == 0:
    print("  => 下载件与 ninfer 产物【同源】：可直接拿去喂 vLLM，无需反拆导出器。")
elif same > 0:
    print("  => 部分一致（%d 同 / %d 不同）：需看不同的那些是哪些张量再定。" % (same, diff))
else:
    print("  => 无一致项：血统不同（或映射仍不对），需检查映射规则。")
