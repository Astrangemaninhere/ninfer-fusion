import sys, os, hashlib
import numpy as np, torch
sys.path.insert(0, r"C:\Users\User\Documents\ziqinzhang\ninfer-fusion-repo")
from tools.artifact.container import Artifact

CK = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ckpts\step_000100.pt"
TUNED = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ckpts\step_000100_tuned.ninfer"
SRC = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
EXP = {  # artifact name -> (state key, shape)  [from patch_dflash2.py MAPPING]
 "dflash2/feature_projection": ("fc.weight",(5120,25600)), "dflash2/context_norm":("hidden_norm.weight",(5120,)),
 "dflash2/final_norm":("norm.weight",(5120,))}
for i in range(5):
    p=f"dflash2/layers/{i}/"; s=f"layers.{i}."
    EXP.update({p+"input_norm":(s+"input_layernorm.weight",(5120,)), p+"post_attention_norm":(s+"post_attention_layernorm.weight",(5120,)),
      p+"attention/query_key_value":(s+"self_attn_qkv.weight",(6144,5120)), p+"attention/context_key":(s+"context_key.weight",(1024,5120)),
      p+"attention/context_value":(s+"context_value.weight",(1024,5120)), p+"attention/query_norm":(s+"self_attn_q_norm.weight",(128,)),
      p+"attention/key_norm":(s+"self_attn_k_norm.weight",(128,)), p+"attention/output":(s+"self_attn_o.weight",(5120,4096)),
      p+"attention_conv/base_kernel":(s+"attention_conv.base",(2,2,5120)), p+"attention_conv/kernel_projection":(s+"attention_conv.proj.weight",(1280,5120)),
      p+"mlp/gate_up":(s+"mlp_gate_up.weight",(34816,5120)), p+"mlp/down":(s+"mlp_down.weight",(5120,17408)),
      p+"mlp_conv/base_kernel":(s+"mlp_conv.base",(2,2,5120)), p+"mlp_conv/kernel_projection":(s+"mlp_conv.proj.weight",(1280,5120))})

st = torch.load(CK, map_location="cpu", weights_only=False)
sd = st["model"] if isinstance(st, dict) and "model" in st else st
def key(k): return "_orig_mod."+k if "_orig_mod."+k in sd else k

art = Artifact.open(TUNED)
names = [o.name for o in art.objects]
missing = [n for n in EXP if n not in names]
print("tuned artifact objects:", len(names), "| dflash2 names missing:", missing)
mismatch = []; nan_tot = 0; inf_tot = 0; checked = 0
for aname,(skey,shape) in EXP.items():
    obj = art.find(aname)
    raw = bytes(art.payload(obj))
    t = sd[key(skey)].detach().to(torch.bfloat16).contiguous()
    if tuple(t.shape) != shape: t = t.view(shape)          # conv bases only
    exp = t.view(torch.uint16).cpu().numpy().tobytes()
    checked += 1
    if len(raw) != len(exp) or hashlib.sha256(raw).digest() != hashlib.sha256(exp).digest():
        mismatch.append(aname); continue
    arr = np.frombuffer(raw, dtype="<u2")
    tf = torch.from_numpy(arr.astype(np.uint16)).view(torch.bfloat16).float()
    nan_tot += int(torch.isnan(tf).sum()); inf_tot += int(torch.isinf(tf).sum())
print(f"round-trip: checked={checked}/73 bit-exact={checked-len(mismatch)} mismatched={mismatch}")
print(f"readback NaN={nan_tot} Inf={inf_tot}")
ds, dt = os.path.getsize(SRC), os.path.getsize(TUNED)
print(f"size: src={ds:,} tuned={dt:,} delta={dt-ds:+,} bytes ({100.0*(dt-ds)/ds:+.4f}%)")
print("VERDICT:", "BIT-EXACT" if not mismatch and not missing and checked==73 and nan_tot==0 and inf_tot==0 else "FLAGGED")
