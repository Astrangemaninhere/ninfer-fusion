#!/usr/bin/env python3
"""对照 artifact 的 Q4 提案头 vs 我们的 bf16 markov_head：
   1) 读 artifact 里 text/draft_head 的对象字段并反量化（Q4G64_F16S：4bit 码 + group64 F16 scale）
   2) torch.load ckpt_0007500.pt，取 [131072,5120] 的那个张量
   3) 量余弦相似度 / 相对误差；并比对我们自己量化回去的误差
"""
import json, struct, pathlib, sys
import torch, numpy as np

ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")
CKPT = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/draft_checkpoints/ckpt_0007500.pt")

with open(ART, "rb") as f:
    f.read(8)
    (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
o = objs["text/draft_head"]
print("text/draft_head:", {k: o.get(k) for k in ("kind", "format", "shape", "bytes")})
rows, cols = o["shape"]
n_elem = rows * cols
nb = o["bytes"]
print("  元素数 = %d  对象字节 = %d" % (n_elem, nb))
print("  若为 4bit码(元素/2) + group64 F16 scale: %d + %d = %d" %
      (n_elem // 2, n_elem // 64 * 2, n_elem // 2 + n_elem // 64 * 2))

with open(ART, "rb") as f:
    f.seek(payload + o["offset"])
    blob = f.read(nb)
codes = np.frombuffer(blob[: n_elem // 2], dtype=np.uint8)
scales = np.frombuffer(blob[n_elem // 2: n_elem // 2 + n_elem // 64 * 2], dtype=np.float16)
print("  codes 形状 = %s  scales 形状 = %s" % (codes.shape, scales.shape))

# 反量化：每个 uint8 里 2 个 4bit 码（低/高 nibble），每 64 元素一个 F16 scale
lo = (codes & 0x0F).astype(np.int16) - 8
hi = ((codes >> 4) & 0x0F).astype(np.int16) - 8
inter = np.empty(n_elem, dtype=np.int16)
inter[0::2] = lo
inter[1::2] = hi
sc = np.repeat(scales.astype(np.float32), 64)
deq = inter.astype(np.float32) * sc
deq = deq.reshape(rows, cols)
print("  反量化完成: mean|.|=%.5f  max|.|=%.4f  (sample row0[:5]=%s)" %
      (np.abs(deq).mean(), np.abs(deq).max(), np.round(deq[0, :5], 4)))

# 读我们的 ckpt
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
keys = list(sd.keys()) if isinstance(sd, dict) else []
print("\nckpt 顶层键 =", keys[:8])
t = None; tname = None
for k, v in (sd.items() if isinstance(sd, dict) else []):
    if isinstance(v, torch.Tensor) and list(v.shape) == [131072, 5120]:
        t, tname = v, k
print("  取到张量:", tname, tuple(t.shape) if t is not None else None)
if t is None:
    sys.exit(2)
mine = t.to(torch.float32).numpy()
a = deq.ravel(); b = mine.ravel()
cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
print("\n=== artifact(Q4) vs ckpt7500(bf16) ===")
print("  cosine = %.6f   相对误差 = %.4f" % (cos, rel))

# 我们自己也量化一遍，看量化误差量级（对称 4bit + group64 F16 scale）
q = mine.reshape(-1, 64)
s = np.abs(q).max(axis=1) / 7.0 + 1e-12
qs = np.clip(np.round(q / s[:, None]), -8, 7)
rec = (qs * s[:, None]).reshape(mine.shape)
relq = float(np.linalg.norm(rec - mine) / (np.linalg.norm(mine) + 1e-12))
print("  我们自己 Q4G64 量化往返的相对误差 = %.4f" % relq)
