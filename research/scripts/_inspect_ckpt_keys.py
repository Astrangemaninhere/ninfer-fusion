import zipfile, pathlib, re, collections

D = pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\draft_checkpoints")
for name in ("ckpt_0007500.pt", "ckpt_0000040.pt"):
    p = D / name
    if not p.exists():
        print("MISSING", name); continue
    print("=" * 70)
    print(name, "%.2f GiB" % (p.stat().st_size / 2**30))
    with zipfile.ZipFile(p) as z:
        names = z.namelist()
        pk = [n for n in names if n.endswith("data.pkl")]
        print("  zip 条目 =", len(names), " data.pkl =", pk[:1])
        if not pk:
            continue
        raw = z.read(pk[0])
        print("  data.pkl 字节 =", len(raw))
        toks = sorted({t.decode() for t in re.findall(rb"[A-Za-z_][A-Za-z0-9_.]{3,70}", raw)})
        interesting = [t for t in toks if any(s in t for s in
                       ("layers.", "context_key", "context_value", "query_key_value", "conv", "norm",
                        "selector", "codebook", "hidden_projection", "fc", "proj", "_orig_mod"))]
        print("  可疑键（前 40）:")
        for t in interesting[:40]:
            print("     " + t)
        print("  总键候选数 =", len(toks))
        # 顶层前缀分布
        pref = collections.Counter(t.split(".")[0] for t in toks)
        print("  顶层前缀:", dict(pref.most_common(8)))
