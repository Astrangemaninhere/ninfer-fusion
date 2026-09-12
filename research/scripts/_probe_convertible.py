import json, os, urllib.request

BASE = "https://hf-mirror.com"
def raw(mid, path="config.json", timeout=30):
    req = urllib.request.Request(f"{BASE}/{mid}/raw/main/{path}", headers={"User-Agent": "p"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

def info(mid, timeout=30):
    req = urllib.request.Request(f"{BASE}/api/models/{mid}?blobs=true", headers={"User-Agent": "p"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())

print("=== A) 选定破禁版 NVFP4 的可转性（转换器要 group_0=FP8 然后 group_1=NVFP4）===")
mid = "sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4"
try:
    c = raw(mid)
    qc = c.get("quantization_config") or {}
    cg = qc.get("config_groups") or {}
    print("  quant_method =", qc.get("quant_method"), " format =", qc.get("format"))
    print("  config_groups 键顺序 =", list(cg.keys()))
    for gname, g in cg.items():
        w = (g or {}).get("weights") or {}
        ia = (g or {}).get("input_activations") or {}
        print(f"    {gname}: weights(format={w.get('format') or g.get('format')}, "
              f"num_bits={w.get('num_bits')}, group={w.get('group_size')}, "
              f"type={w.get('type')}, targets={str(g.get('targets'))[:80]})")
        print(f"      input_activations={ {k: ia.get(k) for k in ('num_bits','type','dynamic','group_size')} }")
    tc = c.get("text_config") or {}
    print("  text_config: layers=%s hidden=%s vocab=%s max_pos=%s" % (
        tc.get("num_hidden_layers"), tc.get("hidden_size"), tc.get("vocab_size"),
        tc.get("max_position_embeddings")))
    print("  architectures =", c.get("architectures"), " model_type =", c.get("model_type"))
except Exception as e:
    print("  失败:", type(e).__name__, e)

print()
print("=== B) 官方 bf16 基座是否可得（转换器硬要求：无 quantization_config + 前端资源哈希匹配）===")
for cand in ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B-Instruct", "Qwen/Qwen3.8-27B-Base"]:
    try:
        d = info(cand)
        sib = d.get("siblings") or []
        tot = sum((s.get("size") or 0) for s in sib)
        shard = sum(1 for s in sib if s["rfilename"].endswith(".safetensors"))
        print(f"  {cand}: 存在 ✓ 文件{len(sib)} 分片{shard} 总 {tot/1e9:.1f} GB gated={d.get('gated')}")
        try:
            c = raw(cand)
            print(f"      has quantization_config = {c.get('quantization_config') is not None}"
                  f"  dtype={ (c.get('text_config') or {}).get('dtype') or c.get('dtype')}")
        except Exception as e2:
            print("      config 读取失败:", e2)
    except Exception as e:
        print(f"  {cand}: 不可得 ({type(e).__name__} {e})")

print()
print("=== C) 已下载进度 ===")
import subprocess
print(subprocess.run(["du", "-sh", "/home/user/models/q38_abl_huihui_nvfp4"],
                     capture_output=True, text=True).stdout.strip())
print(subprocess.run(["bash", "-lc",
                      "ls /home/user/models/q38_abl_huihui_nvfp4 2>/dev/null | head -20"],
                     capture_output=True, text=True).stdout)
