import json, struct, pathlib

CAND = [pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\Qwen3.8-27B"),
        pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2"),
        pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\draft_checkpoints"),
        pathlib.Path(r"C:\Users\User\Documents\ziqinzhang\data\head_export")]
for p in CAND:
    print("=" * 66)
    print(p)
    if not p.exists():
        print("  (不存在)"); continue
    ents = sorted(p.iterdir())
    print("  条目数 =", len(ents))
    for e in ents[:14]:
        print("    %-52s %s" % (e.name, ("%d B" % e.stat().st_size) if e.is_file() else "<dir>"))
    # 找 safetensors 并查 draft_head
    for f in sorted(p.glob("*.safetensors")) + sorted(p.glob("*.json")):
        if f.suffix == ".safetensors":
            try:
                with open(f, "rb") as fh:
                    (n,) = struct.unpack("<Q", fh.read(8))
                    hdr = json.loads(fh.read(n))
                hits = {k: v for k, v in hdr.items() if k != "__metadata__" and ("draft" in k.lower())}
                print("    [%s] 张量 %d, 含 draft 的 %d" % (f.name, len(hdr) - ("__metadata__" in hdr), len(hits)))
                for k, v in list(hits.items())[:6]:
                    print("       %-52s %-8s %s" % (k, v.get("dtype"), str(v.get("shape"))))
            except Exception as ex:
                print("    [%s] 读头失败 %s" % (f.name, ex))
