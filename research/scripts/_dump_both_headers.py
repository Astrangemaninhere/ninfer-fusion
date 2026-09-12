import json, struct, pathlib

ART = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
HF  = r"C:\Users\User\Documents\ziqinzhang\data\draft_dflash2_ref\model.safetensors"

with open(ART, "rb") as fh:
    fh.read(8)
    (jlen,) = struct.unpack("<Q", fh.read(8))
    j = json.loads(fh.read(jlen).decode("utf-8", "replace"))
objs = {o["name"]: o for o in j["objects"]}
print("########## artifact dflash2/* ##########")
print("  (样例字段: %s)" % json.dumps(objs["dflash2/context_norm"], ensure_ascii=False)[:220])
tot = 0
for n in sorted(x for x in objs if x.startswith("dflash2/")):
    o = objs[n]
    print("  %-52s %-10s %-20s bytes=%s" % (n, o.get("dtype"), str(o.get("shape")), o.get("bytes")))
    tot += o.get("bytes", 0)
print("  dflash2 总字节 =", tot)

print()
print("########## HF (incoai) ##########")
with open(HF, "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
keys = sorted(k for k in hdr if k != "__metadata__")
print("  张量数 =", len(keys))
tot2 = 0
for k in keys:
    v = hdr[k]
    sz = v["data_offsets"][1] - v["data_offsets"][0]
    tot2 += sz
    print("  %-56s %-10s %-20s %d" % (k, v.get("dtype"), str(v.get("shape")), sz))
print("  HF 总字节 =", tot2)
