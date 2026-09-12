import torch, glob, os
ck = sorted(glob.glob(r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ckpts\step_*.pt"))[-1]
print("ckpt:", os.path.basename(ck))
st = torch.load(ck, map_location="cpu", weights_only=False)
sd = st["model"] if isinstance(st, dict) and "model" in st else st
sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
print("total keys:", len(sd), "| step:", st.get("step", "?"), "| opt present:", "opt" in st)
classes = {"norm": [], "proj": [], "conv_base": [], "other": []}
for k, v in sd.items():
    if "norm" in k: classes["norm"].append(k)
    elif k.endswith("conv.base"): classes["conv_base"].append(k)
    elif v.ndim == 2: classes["proj"].append(k)
    else: classes["other"].append(k)
bad = 0
for cls, keys in classes.items():
    nan = inf = 0; mx = 0.0; dt = set(); shapes_ok = True
    for k in keys:
        t = sd[k].float()
        n = int(torch.isnan(t).sum()); i = int(torch.isinf(t).sum())
        nan += n; inf += i; mx = max(mx, float(t.abs().max()))
        dt.add(str(sd[k].dtype))
        if n or i: print(f"  FLAG {k}: nan={n} inf={i}")
    bad += nan + inf
    print(f"{cls:10s} n={len(keys):2d} nan={nan} inf={inf} max|w|={mx:.4f} dtypes={sorted(dt)}")
# expected shapes from engine contract
EXP = {"fc.weight": (5120, 25600), "hidden_norm.weight": (5120,), "norm.weight": (5120,)}
for i in range(5):
    p = f"layers.{i}."
    EXP.update({p+"input_layernorm.weight": (5120,), p+"post_attention_layernorm.weight": (5120,),
        p+"self_attn_qkv.weight": (6144, 5120), p+"context_key.weight": (1024, 5120),
        p+"context_value.weight": (1024, 5120), p+"self_attn_q_norm.weight": (128,),
        p+"self_attn_k_norm.weight": (128,), p+"self_attn_o.weight": (5120, 4096),
        p+"attention_conv.base": (2, 2, 320, 16), p+"attention_conv.proj.weight": (1280, 5120),
        p+"mlp_gate_up.weight": (34816, 5120), p+"mlp_down.weight": (5120, 17408),
        p+"mlp_conv.base": (2, 2, 320, 16), p+"mlp_conv.proj.weight": (1280, 5120)})
mism = [(k, tuple(sd[k].shape), e) for k, e in EXP.items() if tuple(sd[k].shape) != e]
print("expected keys:", len(EXP), "| missing:", [k for k in EXP if k not in sd], "| shape mismatches:", mism)
print("VERDICT:", "CLEAN" if bad == 0 and not mism and len(EXP) == len(sd) else "FLAGGED")
