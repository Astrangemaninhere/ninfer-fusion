#!/bin/bash
echo "=== 1) 现有 HF 源目录身份（q3nvfp4）==="
ls /home/user/models/q3nvfp4 2>/dev/null | head -20
echo "--- config.json 关键字段 ---"
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/home/user/models/q3nvfp4/config.json")
if p.exists():
    d = json.loads(p.read_text())
    for k in ("architectures","model_type","quantization_config","torch_dtype","num_hidden_layers","hidden_size","vocab_size"):
        v = d.get(k)
        if k == "quantization_config" and isinstance(v, dict):
            v = {kk: v.get(kk) for kk in ("quant_method","format","config_groups","ignore") if kk in v}
            v = json.dumps(v, ensure_ascii=False)[:400]
        print(f"  {k} = {str(v)[:300]}")
else:
    print("  无 config.json")
PY
echo "--- 是否有 .git 指向 HF repo ---"
cat /home/user/models/q3nvfp4/.git/config 2>/dev/null | head -12
echo "--- README 首行（若有）---"
head -5 /home/user/models/q3nvfp4/README.md 2>/dev/null

echo
echo "=== 2) 磁盘余量（下载 20-28GB 需要）==="
df -h /home/user /mnt/c | head -5

echo
echo "=== 3) 转换器接口（tools/convert）==="
ls /home/user/ninfer-fusion/tools/convert 2>/dev/null | head -30

echo
echo "=== 4) 网络通道：HF 官方 vs 镜像 ==="
timeout 12 curl -sS -o /dev/null -w 'hf.co      http=%{http_code} t=%{time_total}s\n' https://huggingface.co/api/models?limit=1 2>&1
timeout 12 curl -sS -o /dev/null -w 'hf-mirror  http=%{http_code} t=%{time_total}s\n' https://hf-mirror.com/api/models?limit=1 2>&1
echo "HF_ENDPOINT=${HF_ENDPOINT:-<未设>}"
