import json
import re
import collections

TGT = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4-RTX5090"
d = json.load(open(TGT + "/model.safetensors.index.json"))
wm = d["weight_map"]
print("total tensors in index:", len(wm))

layers = collections.Counter()
pat = re.compile(r"\.layers\.(\d+)\.")
for k in wm:
    m = pat.search(k)
    if m:
        layers[int(m.group(1))] += 1
nums = sorted(layers)
print("layer ids present:", nums[0], "..", nums[-1], " count =", len(nums))
miss = [i for i in range(nums[-1] + 1) if i not in layers]
print("missing layer ids:", miss)

top = collections.Counter()
for k in wm:
    top[k.split(".")[0]] += 1
print("top-level prefixes:", dict(top))

# aux layers vLLM will tap for data/draft_model
aux = [1, 14, 29, 44, 57]
print("draft target_layer_ids:", aux)
print("=> vLLM taps 1-based layers:", [i + 1 for i in aux],
      "  all present:", all((i + 1) in layers for i in aux))

for key in ("lm_head.weight", "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight"):
    if key in wm:
        print("FOUND", key, "->", wm[key])

print("\n-- sample keys per shard --")
by = collections.Counter(wm.values())
print(dict(by))
print("\n-- first 12 keys --")
for k in sorted(wm)[:12]:
    print("  ", k)
print("\n-- last 12 keys --")
for k in sorted(wm)[-12:]:
    print("  ", k)
print("\n-- any mtp/nextn tensors? --")
mtp = [k for k in wm if "mtp" in k.lower() or "nextn" in k.lower()]
print("count:", len(mtp), mtp[:5])
print("\n-- linear_attn / self_attn / mlp counts for layer 0 --")
for k in sorted(wm):
    if re.search(r"\.layers\.0\.", k) and "layers.10" not in k:
        print("  ", k)
