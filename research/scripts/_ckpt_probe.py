#!/usr/bin/env python3
"""零内存取证：.pt 是 zip，只读中央目录 + data.pkl（小），比对两炉 ckpt 的键集合与配置痕迹。"""
import zipfile, pathlib, re, collections
D = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/dflash2_ckpts")
CK = ["step_001200.pt", "step_001900.pt", "step_000100.pt", "step_000200.pt"]

def probe(name):
    p = D / name
    if not p.exists():
        return f"{name}: 不存在"
    out = [f"== {name}  size={p.stat().st_size/2**30:.2f} GiB"]
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            out.append(f"   zip 条目 {len(names)}: {names[:6]}")
            # 只读小的 pickle
            pk = [n for n in names if n.endswith("data.pkl")]
            if pk:
                raw = z.read(pk[0])
                out.append(f"   data.pkl {len(raw)} 字节")
                keys = sorted(set(re.findall(rb"([A-Za-z_][A-Za-z0-9_.]{4,60})", raw)))
                kk = [k.decode() for k in keys if any(s in k for s in (b"layers.", b"conv", b"head", b"norm", b"proj", b"embed"))]
                out.append(f"   张量名候选 {len(kk)}: {kk[:8]}")
                # 配置痕迹
                hint = [s.decode() for s in keys if any(h in s.lower() for h in (b"mask", b"shift", b"ctx", b"step", b"args", b"config", b"target"))]
                out.append(f"   配置/元数据痕迹: {hint[:10]}")
                ctr = collections.Counter(len(k) for k in keys)
                out.append(f"   pickle 内名数={len(keys)}")
    except Exception as e:
        out.append(f"   zip 读取失败: {type(e).__name__}: {e}")
    return "\n".join(out)

for c in CK:
    print(probe(c)); print()
