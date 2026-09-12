"""把 incoai 的自洽 dflash2 草稿头（HF safetensors）整体转进 .ninfer artifact。

依据（已字节验证）：
  artifact 76 个 dflash2/* 对象 vs HF 81 个张量：
    query_key_value [6144,5120]  = q_proj[4096,5120] + k_proj[1024,5120] + v_proj[1024,5120]
    mlp/gate_up     [34816,5120] = gate_proj[17408,5120] + up_proj[17408,5120]
    context_key/context_value    = k_proj / v_proj（与 QKV 中同源，字节相同已验证）
    其余 71 个一一对应，形状一致，两边均 BF16

纯 stdlib：读 artifact 头 → 复制底模 → 按 76 对象的 offset 原位写入 → 回读校验。
"""
import json, struct, shutil, hashlib, sys, pathlib, time, re

BASE = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer")
HF   = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\draft_dflash2_ref\model.safetensors")
OUT  = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ref_head.ninfer")

def align_up(x, a=4096):
    return ((x + a - 1) // a) * a

# ---------- 读 artifact 头 ----------
with open(BASE, "rb") as f:
    magic = f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    art = json.loads(f.read(jlen).decode("utf-8", "replace"))
art_payload = align_up(16 + jlen)
objs = {o["name"]: o for o in art["objects"]}
assert magic == b"NINFER\x00\x02", magic

# ---------- 读 HF 头 ----------
with open(HF, "rb") as f:
    (hn,) = struct.unpack("<Q", f.read(8))
    hf = json.loads(f.read(hn))
hf_payload = 8 + hn
hf_keys = {k: v for k, v in hf.items() if k != "__metadata__"}

def hf_off(k):
    o = hf_keys[k]["data_offsets"]
    return hf_payload + o[0], o[1] - o[0]

# ---------- 组装 76 个目标对象的源 ----------
L = 5
MAPPING = {}
for i in range(L):
    b = "layers.%d" % i
    MAPPING["dflash2/layers/%d/input_norm" % i]          = ("single", b + ".input_layernorm.weight")
    MAPPING["dflash2/layers/%d/post_attention_norm" % i] = ("single", b + ".post_attention_layernorm.weight")
    MAPPING["dflash2/layers/%d/attention/query_key_value" % i] = ("concat", [b + ".self_attn.q_proj.weight",
                                                                            b + ".self_attn.k_proj.weight",
                                                                            b + ".self_attn.v_proj.weight"])
    MAPPING["dflash2/layers/%d/attention/context_key" % i]   = ("single", b + ".self_attn.k_proj.weight")
    MAPPING["dflash2/layers/%d/attention/context_value" % i] = ("single", b + ".self_attn.v_proj.weight")
    MAPPING["dflash2/layers/%d/attention/query_norm" % i]    = ("single", b + ".self_attn.q_norm.weight")
    MAPPING["dflash2/layers/%d/attention/key_norm" % i]      = ("single", b + ".self_attn.k_norm.weight")
    MAPPING["dflash2/layers/%d/attention/output" % i]        = ("single", b + ".self_attn.o_proj.weight")
    MAPPING["dflash2/layers/%d/attention_conv/base_kernel" % i]      = ("single", b + ".attention_conv.base_kernel")
    MAPPING["dflash2/layers/%d/attention_conv/kernel_projection" % i] = ("single", b + ".attention_conv.kernel_projection.weight")
    MAPPING["dflash2/layers/%d/mlp/gate_up" % i] = ("concat", [b + ".mlp.gate_proj.weight", b + ".mlp.up_proj.weight"])
    MAPPING["dflash2/layers/%d/mlp/down" % i]    = ("single", b + ".mlp.down_proj.weight")
    MAPPING["dflash2/layers/%d/mlp_conv/base_kernel" % i]       = ("single", b + ".mlp_conv.base_kernel")
    MAPPING["dflash2/layers/%d/mlp_conv/kernel_projection" % i] = ("single", b + ".mlp_conv.kernel_projection.weight")
MAPPING["dflash2/feature_projection"] = ("single", "fc.weight")
MAPPING["dflash2/context_norm"]       = ("single", "hidden_norm.weight")
MAPPING["dflash2/final_norm"]         = ("single", "norm.weight")
MAPPING["dflash2/candidate_selector/hidden_projection"]    = ("single", "candidate_selector.hidden_projection.weight")
MAPPING["dflash2/candidate_selector/predecessor_codebook"] = ("single", "candidate_selector.predecessor_codebook")
MAPPING["dflash2/candidate_selector/successor_codebook"]   = ("single", "candidate_selector.successor_codebook")

targets = [n for n in objs if n.startswith("dflash2/")]
print("artifact dflash2 对象 =", len(targets), " 映射表 =", len(MAPPING))
missing = [n for n in targets if n not in MAPPING]
extra   = [n for n in MAPPING if n not in objs]
print("映射缺项 =", missing)
print("映射多项 =", extra)
if missing or extra:
    sys.exit(2)

# 预检查所有源键存在 + 逐对象字节数吻合
plan = []   # (name, offset, expected_bytes, [(hf_key, off, size), ...])
for n in targets:
    kind, src = MAPPING[n]
    keys = [src] if kind == "single" else src
    parts = []
    for k in keys:
        if k not in hf_keys:
            print("HF 缺键:", k, " (for", n, ")"); sys.exit(3)
        parts.append((k,) + hf_off(k))
    total = sum(p[2] for p in parts)
    exp = objs[n]["bytes"]
    if total != exp:
        print("字节不符: %s  artifact=%d  hf=%d" % (n, exp, total)); sys.exit(4)
    plan.append((n, objs[n]["offset"], exp, parts))
print("全部 76 个对象字节数吻合")

# ---------- 复制底模 → 原位写入 ----------
print("复制底模 ->", OUT.name, "...")
t0 = time.time()
shutil.copyfile(BASE, OUT)
print("  复制完成 %.1fs, 大小 %d" % (time.time()-t0, OUT.stat().st_size))

with open(HF, "rb") as fs, open(OUT, "r+b") as fo:
    written = 0
    for n, off, exp, parts in plan:
        fo.seek(art_payload + off)
        for (k, so, ss) in parts:
            fs.seek(so)
            buf = fs.read(ss)
            assert len(buf) == ss
            fo.write(buf)
        written += exp
    fo.flush()
print("写入字节 =", written)

# ---------- 回读校验（抽 6 个对象，比对 HF 源） ----------
print("回读校验：")
def md5f(b): return hashlib.md5(b).hexdigest()[:16]
check = ["dflash2/candidate_selector/predecessor_codebook",
         "dflash2/feature_projection",
         "dflash2/layers/0/attention/query_key_value",
         "dflash2/layers/4/mlp/gate_up",
         "dflash2/context_norm",
         "dflash2/layers/2/attention/context_value"]
with open(HF, "rb") as fs, open(OUT, "rb") as fo:
    for n in check:
        o = objs[n]
        fo.seek(art_payload + o["offset"]); got = fo.read(o["bytes"])
        kind, src = MAPPING[n]
        keys = [src] if kind == "single" else src
        want = b""
        for k in keys:
            so, ss = hf_off(k); fs.seek(so); want += fs.read(ss)
        ok = md5f(got) == md5f(want)
        print("  %-56s %s  %s" % (n, md5f(got), "MATCH" if ok else "MISMATCH"))
print("OUT =", OUT)
print("CONVERT_DONE")
