import json, struct, hashlib

ART = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
with open(ART, "rb") as fh:
    fh.read(8)
    (jlen,) = struct.unpack("<Q", fh.read(8))
    j = json.loads(fh.read(jlen).decode("utf-8", "replace"))
objs = {o["name"]: o for o in j["objects"]}

# payload 起点：align_up(16 + jlen, 4096)
import math
payload_start = ((16 + jlen + 4095) // 4096) * 4096
print("json_len =", jlen, " payload_start =", payload_start)

def read_obj(name):
    o = objs[name]
    off = o["offset"]; nb = o["bytes"]
    with open(ART, "rb") as f:
        f.seek(payload_start + off)
        return f.read(nb)

def md5b(b):
    return hashlib.md5(b).hexdigest()[:16]

qkv = read_obj("dflash2/layers/0/attention/query_key_value")   # [6144,5120] BF16 -> 每行 10240 B
row = 5120 * 2
q = qkv[0:4096*row]
k = qkv[4096*row:5120*row]
v = qkv[5120*row:6144*row]
ck = read_obj("dflash2/layers/0/attention/context_key")
cv = read_obj("dflash2/layers/0/attention/context_value")
print("layer0: qkv bytes=%d  q=%d k=%d v=%d" % (len(qkv), len(q), len(k), len(v)))
print("  md5 k   =", md5b(k))
print("  md5 ck  =", md5b(ck), "  k == context_key ?", k == ck)
print("  md5 v   =", md5b(v))
print("  md5 cv  =", md5b(cv), "  v == context_value ?", v == cv)

# gate_up 也验一下拆分与 mlp/down 无关，仅确认形状逻辑
gu = read_obj("dflash2/layers/0/mlp/gate_up")
print("layer0: gate_up bytes=%d (期望 34816*5120*2=%d) ->" % (len(gu), 34816*5120*2), len(gu) == 34816*5120*2)

# 顺便看 dflash2 之外的 tensor 里是否有 selector 的重复副本（避免我误判命名空间）
names = [o["name"] for o in j["objects"] if o["kind"] == "tensor"]
print("tensor 总数 =", len(names), " 含 'selector' 的：", [n for n in names if "selector" in n][:8])
print("含 'dflash2' 的 tensor 数 =", len([n for n in names if n.startswith("dflash2/")]))
