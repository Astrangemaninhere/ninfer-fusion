#!/bin/bash
set -u
R=/home/user/ninfer-fusion
J=/mnt/c/Users/User/Documents/ziqinzhang
P='请用中文写一段两百字左右的短文，介绍西湖一年四季的景色变化，要求语句连贯、不要列条目。'
cd "$R/build" || exit 3
exec > >(tee -a "$J/dl/feat2.log") 2>&1
echo "=== dump $(date '+%H:%M:%S') ==="
rm -f $J/dl/feat_*.bin
NINFER_DF2FEAT=1 timeout 900 ./apps/ninfer /home/user/models/qwen3_8_27b_nvfp4_dflash2.ninfer \
  --prompt "$P" --max-new 16 --max-context 4096 --no-thinking --greedy --spec dflash2 \
  > $J/dl/feat_stdout2.log 2>&1
echo "  rc=$?"; grep -E 'df2feat' $J/dl/feat_stdout2.log | head -2
ls -l $J/dl/feat_*.bin | awk '{print "  ", $5, $9}'

echo
echo "=== 离线复算 ==="
/home/user/vllm029/bin/python - <<'PY'
import numpy as np, json, struct, pathlib
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/dl")
ART = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2/qwen3_8_27b_nvfp4_dflash2.ninfer")

feat = np.fromfile(J/"feat_features.bin", dtype=np.float16).astype(np.float32)
proj = np.fromfile(J/"feat_projected.bin", dtype=np.float16).astype(np.float32)
cont = np.fromfile(J/"feat_context.bin", dtype=np.float16).astype(np.float32)
print("  features=%d projected=%d context=%d" % (feat.size, proj.size, cont.size))
F, C = 25600, 8
P_, C2 = 5120, 8
feat = feat.reshape(F, C); proj = proj.reshape(P_, C2); cont = cont.reshape(P_, C2)

# 读 artifact 的 fc / context_norm
with open(ART, "rb") as f:
    f.read(8); (jlen,) = struct.unpack("<Q", f.read(8))
    j = json.loads(f.read(jlen).decode("utf-8","replace"))
payload = ((16+jlen+4095)//4096)*4096
objs = {o["name"]: o for o in j["objects"]}
def read_obj(name):
    o = objs[name]
    with open(ART, "rb") as f:
        f.seek(payload + o["offset"]); return f.read(o["bytes"])
fc = np.frombuffer(read_obj("dflash2/feature_projection"), dtype=np.float16).astype(np.float32).reshape(5120, 25600)
cn = np.frombuffer(read_obj("dflash2/context_norm"), dtype=np.float16).astype(np.float32).reshape(5120)
print("  fc", fc.shape, " context_norm", cn.shape)

exp_proj = fc @ feat
rel = np.linalg.norm(exp_proj - proj) / (np.linalg.norm(proj) + 1e-9)
print("  [阶段1] projected = fc @ features : 相对误差 = %.5f" % rel)

# rmsnorm（无 weight 前/后）：context = rmsnorm(projected) * norm_weight
ms = (proj ** 2).mean(axis=0, keepdims=True)
exp_cont = (proj / np.sqrt(ms + 1e-6)) * cn[:, None]
rel2 = np.linalg.norm(exp_cont - cont) / (np.linalg.norm(cont) + 1e-9)
print("  [阶段2] context   = rmsnorm(projected)*w : 相对误差 = %.5f" % rel2)

print("  [tap 检查] features 的 5 段（各 5120 行）:")
for i in range(5):
    seg = feat[i*5120:(i+1)*5120]
    print("    tap%d: norm=%.3f mean=%.5f std=%.5f max=%.3f" %
          (i, np.linalg.norm(seg), seg.mean(), seg.std(), np.abs(seg).max()))
# tap 之间的余弦相似度（若某两段几乎相同 => 取重复层）
def cos(a, b): return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
print("    tap 两两余弦:")
for i in range(5):
    row = "    ".join("%.3f" % cos(feat[i*5120:(i+1)*5120], feat[k*5120:(k+1)*5120]) for k in range(5))
    print("      tap%d: %s" % (i, row))
PY
echo FEAT2_DONE
