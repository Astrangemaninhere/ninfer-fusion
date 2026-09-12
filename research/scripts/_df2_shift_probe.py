#!/usr/bin/env python3
"""Which target does the TRAINED draft actually match: ids16[a+i] or ids16[a+1+i]?

Training (train_dflash2.py:421-427) feeds block = tokens[a:a+B] and pairs the
model's block position i with ids16[a+1+i]. Under the measured dump convention
(ids16[t] is the teacher's distribution for tokens[t+1]), the model at block
position i predicts tokens[a+i+1], whose teacher row is ids16[a+i] — so the code
would be shifted one position forward. This probe settles it on the real
checkpoint with a few forward passes: whichever row the draft's top-1 matches
better is the convention it was trained under.
"""
import glob
import json
import sys

import numpy as np
import torch

WORK = r"C:\Users\User\Documents\ziqinzhang"
CACHE = WORK + r"\data\hs_cache_topk2"
CKPT = sys.argv[1] if len(sys.argv) > 1 else WORK + r"\data\dflash2_ckpts\step_001200.pt"
TEACHER = WORK + r"\data\Qwen3.8-27B"
CTX = 128                      # matches training's --max-ctx 128
N_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 6

sys.path.insert(0, WORK)
import train_dflash2 as T  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

head = T.DFlash2Head().to(device).to(torch.bfloat16)
st = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
sd = st["model"] if "model" in st else st
# torch.compile leaves `_orig_mod.` prefixes; strip them inline (the helper lives
# in eval_ddtree.py:98, not in the trainer).
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
head.load_state_dict(sd)
head.eval()
print("loaded %d tensors from %s" % (len(sd), CKPT))

from safetensors import safe_open  # noqa: E402
with open(TEACHER + r"\model.safetensors.index.json") as f:
    wmap = json.load(f)["weight_map"]
lm_file = wmap.get("lm_head.weight") or wmap.get("model.language_model.lm_head.weight")
emb_file = (wmap.get("model.embed_tokens.weight")
            or wmap.get("model.language_model.embed_tokens.weight"))
with safe_open(TEACHER + "\\" + lm_file, framework="pt", device="cpu") as sf:
    lm_w = sf.get_tensor([k for k in sf.keys() if "lm_head" in k][0]).float()
with safe_open(TEACHER + "\\" + emb_file, framework="pt", device="cpu") as sf:
    emb_w = sf.get_tensor([k for k in sf.keys() if "embed" in k.lower()
                           and "weight" in k][0]).float()
lm_head = torch.nn.Linear(T.HIDDEN, T.VOCAB, bias=False).to(device)
lm_head.weight.data = lm_w.nan_to_num().to(torch.bfloat16).to(device)
embed = torch.nn.Embedding(T.VOCAB, T.HIDDEN).to(device)
embed.weight.data = emb_w.nan_to_num().to(torch.bfloat16).to(device)
for p in list(lm_head.parameters()) + list(embed.parameters()):
    p.requires_grad_(False)

B = T.BLOCK_SIZE
aligned = shifted = total = 0
rows = []
for path in sorted(glob.glob(CACHE + r"\seq_*.npz"))[:N_FILES]:
    # use the trainer's own loader so the feature packing is identical to training
    tok, feat, _last, _plen, ids16, _v16 = T.load_cache_seq(path, device)
    if ids16 is None:
        continue
    anchors = [x for x in range(CTX, min(len(tok) - B, CTX + 24 * 8), 8)
               if x + B <= len(tok)]
    for a in anchors:
        c_lo = max(0, a - CTX)
        ctx = feat[c_lo:a].unsqueeze(0)
        ctx_pos = torch.arange(c_lo, a, device=device).unsqueeze(0)
        block = tok[a:a + B].unsqueeze(0)
        block_pos = torch.arange(a, a + B, device=device).unsqueeze(0)
        with torch.no_grad():
            dlogits, _ = head(ctx.to(torch.bfloat16), block, lm_head, embed, ctx_pos,
                              block_pos)
        top1 = dlogits.argmax(dim=-1)[0]          # [B] top-1 per block position
        for i in range(B - 1):
            a_ok = int(top1[i].item() == int(ids16[a + i, 0]))       # aligned
            s_ok = int(top1[i].item() == int(ids16[a + 1 + i, 0]))   # as trained
            aligned += a_ok
            shifted += s_ok
            total += 1
            if i < 3:
                rows.append("    pos %d: top1=%d  ids16[a+i]=%d(%s)  ids16[a+1+i]=%d(%s)"
                            % (i, top1[i].item(), int(ids16[a + i, 0]),
                               "hit" if a_ok else "-", int(ids16[a + 1 + i, 0]),
                               "hit" if s_ok else "-"))

print("\n".join(rows[:6]))
print()
ra = aligned / max(1, total)
rs = shifted / max(1, total)
print("block-position top-1 agreement over %d (anchor, position) pairs:" % total)
print("  vs ids16[a+i]   (aligned : what the next-token convention implies) : %.4f" % ra)
print("  vs ids16[a+1+i] (SHIFTED : what train_dflash2.py used to train)    : %.4f" % rs)
print()
# NOTE: compare RATES, not counts (an earlier version of this probe compared the
# integer counts and printed a verdict the numbers did not support).
if rs > ra + 0.05:
    print("VERDICT: SHIFTED WINS -> training target was off by one (mechanism bug)")
elif ra > rs + 0.05:
    print("VERDICT: ALIGNED WINS -> training target consistent; look elsewhere")
else:
    print("VERDICT: ambiguous (%.4f vs %.4f, gap %.4f) -> need the A/B arms"
          % (ra, rs, abs(ra - rs)))
