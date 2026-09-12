import numpy as np, json, struct, pathlib

J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")

print("=== 1) 按 ne[0] 连续轴正确读 features ===")
for call in (5, 6, 7, 8):
    f = J / ("feat_features_%d.bin" % call)
    if not f.exists(): continue
    raw = np.fromfile(f, dtype=np.float16).astype(np.float32)
    n = raw.size
    # [25600, 8] 且 ne[0] 连续 => 内存序为 (col, row) => reshape(8, 25600)
    v = raw.reshape(8, 25600)
    col0 = v[0]
    taps = [float(np.linalg.norm(col0[i*5120:(i+1)*5120])) for i in range(5)]
    coln = [float(np.linalg.norm(v[c])) for c in range(8)]
    print("  call=%d 总元素=%d" % (call, n))
    print("    各列范数 = %s" % ["%.2f" % x for x in coln])
    print("    第0列 5 个 tap 范数 = %s  (期望随深度递增)" % ["%.2f" % x for x in taps])

print()
print("=== 2) 补上被作废的算术验证（用正确轴向） ===")
feat = np.fromfile(J / "feat_features_5.bin", dtype=np.float16).astype(np.float32).reshape(8, 25600)
proj = np.fromfile(J / "feat_projected_5.bin", dtype=np.float16).astype(np.float32).reshape(8, 5120)
cont = np.fromfile(J / "feat_context_5.bin", dtype=np.float16).astype(np.float32).reshape(8, 5120)
print("  shapes:", feat.shape, proj.shape, cont.shape)

with open(ART, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8", "replace"))
payload = ((16 + jlen + 4095) // 4096) * 4096
objs = {o["name"]: o for o in j["objects"]}
def read_obj(name):
    o = objs[name]
    with open(ART, "rb") as f:
        f.seek(payload + o["offset"]); return f.read(o["bytes"])
fc = np.frombuffer(read_obj("dflash2/feature_projection"), dtype=np.float16).astype(np.float32).reshape(5120, 25600)
cn = np.frombuffer(read_obj("dflash2/context_norm"), dtype=np.float16).astype(np.float32).reshape(5120)

# projected[c] = fc @ features[c]  （features 第 c 列是 25600 维）
exp_proj = (fc @ feat.T).T
den = np.linalg.norm(proj) + 1e-9
rel = np.linalg.norm(exp_proj - proj) / den
print("  [阶段1] projected = fc @ features       相对误差 = %.6f" % rel)
ms = (proj ** 2).mean(axis=1, keepdims=True)
exp_cont = (proj / np.sqrt(ms + 1e-6)) * cn[None, :]
rel2 = np.linalg.norm(exp_cont - cont) / (np.linalg.norm(cont) + 1e-9)
print("  [阶段2] context = rmsnorm(projected)*w   相对误差 = %.6f" % rel2)
print("  注意：仅第 0 列有真实数据，其余列为零，故相对误差主要由第 0 列决定")
