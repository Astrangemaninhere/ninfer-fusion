import json
p = "/mnt/c/Users/User/Documents/ziqinzhang/models/Qwen3.8-27B-NVFP4/config.json"
c = json.load(open(p))
print("arch      =", c.get("architectures"), "| model_type =", c.get("model_type"))
print("auto_map  =", c.get("auto_map"))
print("lmo       =", c.get("language_model_only"))
t = c.get("text_config", c)
print("layers    =", t.get("num_hidden_layers"), "hidden =", t.get("hidden_size"),
      "vocab =", t.get("vocab_size"))
print("dtype     =", c.get("dtype"), "/", t.get("dtype"))
v = c.get("vision_config") or {}
print("vision    =", v.get("num_hidden_layers") or v.get("depth"), v.get("hidden_size"))
q = c.get("quantization_config") or {}
print("quant keys=", list(q.keys()))
print("quant fmt =", q.get("format"), "quant_method =", q.get("quant_method"))
print("cgroups   =", list((q.get("config_groups") or {}).keys()))
for k, g in (q.get("config_groups") or {}).items():
    print("   ", k, "w=", (g.get("weights") or {}).get("num_bits"),
          (g.get("weights") or {}).get("type"),
          "gs=", (g.get("weights") or {}).get("group_size"),
          "a=", (g.get("input_activations") or {}).get("num_bits"),
          (g.get("input_activations") or {}).get("type"),
          "ntargets=", len(g.get("targets") or []))
print("n_ignore  =", len(q.get("ignore") or []))
# layer_types sanity
lt = t.get("layer_types") or []
print("layer_types len =", len(lt), "full_attn idx =",
      [i for i, x in enumerate(lt) if x == "full_attention"][:8], "...")
