#!/usr/bin/env python3
"""Which target row does the TRAINED draft's block-position top-1 actually match?

The trainer historically paired block position i with ids16[a+1+i] (--target-shift 1)
and can pair it with ids16[a+i] (--target-shift 0). The engine's verify compares
drafts[j] against the target's argmax at verify column j while drafts[j] sits at
verify column j+1 -- i.e. the draft at block column j+1 must emit the token that
BELONGS at position frontier+j+1 (its own masked slot), whose teacher row is
ids16[frontier+j] (ids16[t] = teacher distribution for tokens[t+1]).

So three arms are possible, not two:
    row ids16[a+i-1]   "own-slot / masked"   = what the engine's verify demands
    row ids16[a+i]     "next-token"
    row ids16[a+1+i]   "legacy trainer default"
Whichever arm wins tells us the effective training convention; comparing that with
the engine's demand decides whether the mismatch is in training or in the engine.
"""
import glob
import json
import sys

import torch

WORK = r"C:\Users\User\Documents\ziqinzhang"
CACHE = WORK + r"\data\hs_cache_topk2"
CKPT = sys.argv[1] if len(sys.argv) > 1 else WORK + r"\data\dflash2_ckpts\step_001200.pt"
TEACHER = WORK + r"\data\Qwen3.8-27B"
CTX = 128
N_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 3

sys.path.insert(0, WORK)
import train_dflash2 as T  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, flush=True)

head = T.DFlash2Head().to(device).to(torch.bfloat16)
st = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
sd = st["model"] if "model" in st else st
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
head.load_state_dict(sd)
head.eval()
print("loaded %d tensors from %s" % (len(sd), CKPT), flush=True)
del st

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
del lm_w, emb_w
print("teacher head/embed on %s" % device, flush=True)

B = T.BLOCK_SIZE
cnt = {"own": 0, "next": 0, "legacy": 0}
total = 0
samples = []
for path in sorted(glob.glob(CACHE + r"\seq_*.npz"))[:N_FILES]:
    tok, feat, _last, _plen, ids16, _v16 = T.load_cache_seq(path, device)
    if ids16 is None:
        continue
    anchors = [x for x in range(CTX, min(len(tok) - B, CTX + 24 * 8), 8) if x + B <= len(tok)]
    print("  %s: %d anchors" % (path.rsplit("\\", 1)[-1], len(anchors)), flush=True)
    for a in anchors:
        c_lo = max(0, a - CTX)
        ctx = feat[c_lo:a].unsqueeze(0)
        ctx_pos = torch.arange(c_lo, a, device=device).unsqueeze(0)
        block = tok[a:a + B].unsqueeze(0)
        block_pos = torch.arange(a, a + B, device=device).unsqueeze(0)
        with torch.no_grad():
            dlogits, _ = head(ctx.to(torch.bfloat16), block, lm_head, embed, ctx_pos, block_pos)
        top1 = dlogits.argmax(dim=-1)[0]
        for i in range(B - 1):
            t1 = int(top1[i].item())
            own = int(ids16[a + i - 1, 0]) if a + i - 1 >= 0 else -1
            nxt = int(ids16[a + i, 0])
            leg = int(ids16[a + 1 + i, 0])
            cnt["own"] += int(t1 == own)
            cnt["next"] += int(t1 == nxt)
            cnt["legacy"] += int(t1 == leg)
            total += 1
            if len(samples) < 6:
                samples.append("    pos %d top1=%d | own[a+i-1]=%d%s | next[a+i]=%d%s | legacy[a+1+i]=%d%s"
                               % (i, t1, own, "*" if t1 == own else " ",
                                  nxt, "*" if t1 == nxt else " ",
                                  leg, "*" if t1 == leg else " "))
print("\n".join(samples))
print()
print("block-position top-1 agreement over %d (anchor, position) pairs:" % total)
for k, label in (("own", "ids16[a+i-1] own-slot/masked  (what the engine's verify demands)"),
                 ("next", "ids16[a+i]   next-token"),
                 ("legacy", "ids16[a+1+i] legacy trainer default")):
    print("  %-58s %.4f" % (label, cnt[k] / max(1, total)))
best = max(cnt, key=lambda k: cnt[k])
print()
print("VERDICT: best arm = %s (%.4f)" % (best, cnt[best] / max(1, total)))
if best == "own":
    print("  -> draft already matches the engine's convention; a low engine acceptance")
    print("     then points at the engine/draft plumbing, not at a training shift.")
elif best == "legacy":
    print("  -> the checkpoint learned the two-ahead convention: retrain with")
    print("     --target-shift 0 (and re-run this probe) before blaming the engine.")
else:
    print("  -> the checkpoint learned next-token; the engine's verify wants own-slot,")
    print("     so either the mask/block construction or the accept pairing is off by one.")
