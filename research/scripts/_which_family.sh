#!/bin/bash
J=/mnt/c/Users/User/Documents/ziqinzhang
echo "=== 1) train_dflash2.py 里定义的模块/键名 ==="
grep -nE 'class .*\(|self\.[a-z_]+ *=|markov|qkv_proj|attention_conv|context_key|codebook|selector|head' $J/train_dflash2.py 2>/dev/null | head -40
echo
echo "=== 2) 是否有 dspark 的训练/转换脚本 ==="
ls -1 $J/train*.py $J/*dspark* 2>/dev/null | head
ls -1 $J/ninfer-fusion-repo/tools/convert/qwen3_8_27b/ 2>/dev/null | head -12
echo
echo "=== 3) dflash2_ckpts 与 draft_checkpoints 的键族对照（看 pickle 文本） ==="
python3 - <<'PY'
import zipfile, re, pathlib
for p in [pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/dflash2_ckpts/step_001200.pt"),
          pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/data/draft_checkpoints/ckpt_0007500.pt")]:
    if not p.exists(): print("MISSING", p); continue
    with zipfile.ZipFile(p) as z:
        pk = [n for n in z.namelist() if n.endswith("data.pkl")]
        raw = z.read(pk[0]) if pk else b""
    toks = sorted({t.decode() for t in re.findall(rb"[A-Za-z_][A-Za-z0-9_.]{3,60}", raw)})
    fam = {
        "attention_conv": sum("attention_conv" in t for t in toks),
        "mlp_conv":       sum("mlp_conv" in t for t in toks),
        "context_key":    sum("context_key" in t for t in toks),
        "qkv_proj":       sum("qkv_proj" in t for t in toks),
        "markov_head":    sum("markov_head" in t for t in toks),
        "codebook":       sum("codebook" in t for t in toks),
        "conv.base":      sum("conv.base" in t for t in toks),
    }
    print(p.name, " 键族计数:", fam)
PY
