#!/usr/bin/env python3
"""Z1 recon: inventory candidate targets/drafts for vLLM DSpark acceptance run."""
import json, os, glob, sys

ROOT = "/mnt/c/Users/User/Documents/ziqinzhang"

def brief_cfg(p):
    try:
        c = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        return "  <unreadable: %s>" % e
    out = []
    out.append("  architectures = %s" % c.get("architectures"))
    out.append("  model_type    = %s" % c.get("model_type"))
    t = c.get("text_config", c)
    for k in ("num_hidden_layers", "hidden_size", "vocab_size", "head_dim",
              "num_attention_heads", "num_key_value_heads",
              "full_attention_interval", "tie_word_embeddings", "dtype",
              "block_size", "markov_rank", "num_target_layers",
              "target_layer_ids", "num_anchors", "mask_token_id",
              "enable_confidence_head", "n_predict", "num_nextn_predict_layers"):
        if k in t:
            out.append("  text.%-26s = %s" % (k, t[k]))
        elif k in c:
            out.append("  top.%-27s = %s" % (k, c[k]))
    q = c.get("quantization_config") or t.get("quantization_config")
    if q:
        out.append("  quantization_config = %s" % json.dumps(q)[:400])
    return "\n".join(out)

def sf_index(d):
    """Return (n_shards, list_of_shard_names, index_json_or_None)."""
    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    idx = os.path.join(d, "model.safetensors.index.json")
    ij = None
    if os.path.exists(idx):
        try:
            ij = json.load(open(idx, encoding="utf-8"))
        except Exception:
            pass
    return shards, ij

print("=" * 78)
print("MODELS DIR SCAN")
print("=" * 78)
for d in sorted(glob.glob(os.path.join(ROOT, "models", "*"))):
    if not os.path.isdir(d):
        continue
    name = os.path.basename(d)
    cfg = os.path.join(d, "config.json")
    shards, ij = sf_index(d)
    other = [f for f in os.listdir(d)
             if f.endswith((".ninfer", ".gguf", ".bin"))]
    print("\n### %s" % name)
    print("  config.json : %s" % ("YES" if os.path.exists(cfg) else "MISSING"))
    print("  shards(%d)   : %s" % (len(shards), [os.path.basename(s) for s in shards][:6]))
    if ij:
        wm = ij.get("weight_map", {})
        print("  index: %d tensors" % len(wm))
        print("  index sample: %s" % list(wm.items())[:4])
    if other:
        print("  OTHER       : %s" % [(f, os.path.getsize(os.path.join(d, f))) for f in other][:5])
    if os.path.exists(cfg):
        print(brief_cfg(cfg))

print()
print("=" * 78)
print("DRAFT_MODEL")
print("=" * 78)
d = os.path.join(ROOT, "data", "draft_model")
print("files:", [(f, os.path.getsize(os.path.join(d, f))) for f in os.listdir(d)])
print(brief_cfg(os.path.join(d, "config.json")))

print()
print("=" * 78)
print("_fp8_target")
print("=" * 78)
print(brief_cfg(os.path.join(ROOT, "_fp8_target", "config.json")))
