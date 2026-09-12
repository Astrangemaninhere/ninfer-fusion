import json, struct, pathlib

paths = [r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer",
         r"C:\Users\User\Documents\ziqinzhang\data\dflash2_ckpts\step_000200_tuned.ninfer"]
for p in paths:
    f = pathlib.Path(p)
    if not f.exists():
        print("MISSING:", p); continue
    with open(f, "rb") as fh:
        magic = fh.read(8)
        (jlen,) = struct.unpack("<Q", fh.read(8))
        raw = fh.read(jlen)
    print("=" * 70)
    print(f.name)
    print("  magic =", magic, " json_len =", jlen)
    try:
        j = json.loads(raw.decode("utf-8", "replace"))
    except Exception as e:
        print("  JSON parse failed:", e); continue
    print("  top-level keys:", list(j.keys())[:12])
    objs = j.get("objects") or j.get("tensors") or j.get("entries") or []
    if isinstance(objs, dict):
        names = list(objs.keys())
    elif isinstance(objs, list):
        names = [o.get("name", "?") if isinstance(o, dict) else str(o) for o in objs]
    else:
        names = []
    print("  object count =", len(names))
    df2 = [n for n in names if "dflash2" in n]
    print("  dflash2/* count =", len(df2))
    for n in df2[:60]:
        print("    " + n)
    sel = [n for n in names if ("selector" in n) or ("codebook" in n) or ("candidate" in n)]
    print("  selector-ish count =", len(sel))
    for n in sel[:20]:
        print("    " + n)
