#!/usr/bin/env python3
"""零反量化同源判定：比 ninfer 产物里的 BF16 张量 vs 下载的 HF safetensors。
纯 python 解析 safetensors（8B 头长 + JSON + 数据），不依赖 safetensors 库。
"""
import json, pathlib, struct, hashlib

NINFER = "/home/user/models/qwen3_8_27b_nvfp4.ninfer"
HF_DIR = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090")

# --- 1) 读 ninfer 清单，挑出 BF16 小张量 ---
with open(NINFER, "rb") as f:
    f.read(8); n = struct.unpack("<Q", f.read(8))[0]
    man = json.loads(f.read(n).decode("utf-8", "replace"))
objs = {o["name"]: o for o in man["objects"] if o.get("kind") == "tensor"}
bf16 = {k: v for k, v in objs.items()
        if v.get("format") == "BF16" and v.get("layout") == "contiguous-le-v1"}
print("ninfer BF16/contiguous 张量: %d 个" % len(bf16))
small = sorted(bf16.items(), key=lambda kv: kv[1]["bytes"])[:200]
print("最小的几个:", [(k, v["bytes"]) for k, v in small[:5]])

def read_ninfer(name):
    o = objs[name]
    with open(NINFER, "rb") as f:
        f.seek(o["offset"]); return f.read(o["bytes"])

# --- 2) 解析 HF safetensors 头，建 名字->(文件,偏移,长度) 表 ---
idx = json.loads((HF_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
weight_map = idx["weight_map"]
hdr_cache = {}
def hf_locate(name):
    shard = weight_map.get(name)
    if not shard: return None
    if shard not in hdr_cache:
        p = HF_DIR / shard
        with open(p, "rb") as f:
            hn = struct.unpack("<Q", f.read(8))[0]
            hdr_cache[shard] = (json.loads(f.read(hn).decode("utf-8")), hn, p)
    h, hn, p = hdr_cache[shard]
    e = h.get(name)
    if not e: return None
    return (p, 8 + hn + e["data_offsets"][0], e["data_offsets"][1] - e["data_offsets"][0])

print("\nHF 张量总数: %d" % len(weight_map))
print("HF 里含 'layernorm' 的名字样例: %s" % [x for x in list(weight_map)[:0] or
      [x for x in weight_map if 'norm' in x][:4]])

def read_hf(name):
    loc = hf_locate(name)
    if not loc: return None
    p, off, ln = loc
    with open(p, "rb") as f:
        f.seek(off); return f.read(ln)

# --- 3) 试探性名字映射：ninfer -> HF ---
def candidates(ninfer_name):
    """给几个可能的 HF 名（不同约定）"""
    parts = ninfer_name.split("/")
    if len(parts) >= 3 and parts[1] == "layers":
        L, rest = parts[2], "/".join(parts[3:])
        yield "model.layers.%s.%s.weight" % (L, rest.replace("/", "."))
        yield "model.layers.%s.%s" % (L, rest.replace("/", "."))
        yield "layers.%s.%s.weight" % (L, rest.replace("/", "."))
    elif ninfer_name == "text/token_embedding":
        for c in ("model.embed_tokens.weight", "model.language_model.embed_tokens.weight"):
            yield c
    return

hits = 0; mism = 0; checked = 0
for name in list(bf16)[:400]:
    mine = read_ninfer(name)
    for cand in candidates(name):
        their = read_hf(cand)
        if their is None: continue
        checked += 1
        if len(their) != len(mine):
            print("  尺寸不同: %s (%d) vs %s (%d)" % (name, len(mine), cand, len(their)))
            break
        if their == mine:
            hits += 1
            print("  [一致] %s  ==  %s" % (name, cand))
        else:
            mism += 1
            print("  [不同] %s  !=  %s   (ninfer md5=%s  hf md5=%s)" %
                  (name, cand, hashlib.md5(mine).hexdigest()[:10], hashlib.md5(their).hexdigest()[:10]))
        break
    if checked >= 12: break

print("\n结果: 检查 %d 项, 一致 %d, 不同 %d" % (checked, hits, mism))
print("判读: 有「一致」=> 下载件与 ninfer 同源，可省掉导出器；全部「不同」=> 血统不同，必须反拆。")
