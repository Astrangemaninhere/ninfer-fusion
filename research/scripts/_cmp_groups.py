import json, pathlib
for p in ['/home/user/models/q3nvfp4/config.json',
          '/home/user/models/q38_abl_huihui_nvfp4/config.json']:
    f = pathlib.Path(p)
    print("=" * 20, p, "存在" if f.exists() else "不存在")
    if not f.exists():
        continue
    c = json.loads(f.read_text())
    qc = c.get("quantization_config") or {}
    print("  quant_method =", qc.get("quant_method"), " format =", qc.get("format"))
    cg = qc.get("config_groups") or {}
    print("  config_groups 顺序 =", list(cg.keys()))
    for g, v in cg.items():
        v = v or {}
        w = v.get("weights") or {}
        ia = v.get("input_activations") or {}
        print("   ", g, "weights =", {k: w.get(k) for k in ("format", "num_bits", "group_size", "type")})
        print("        targets =", str(v.get("targets"))[:140])
        print("        input_activations =", {k: ia.get(k) for k in ("format", "num_bits", "dynamic", "group_size", "type")})
    ig = qc.get("ignore") or []
    print("  ignore 条目 =", len(ig), " 含 lm_head =", any("lm_head" in str(x) for x in ig),
          " 含 mtp =", any("mtp" in str(x) for x in ig))
