# -*- coding: utf-8 -*-
"""model_import 自测: 构造最小合法 GGUF/HF fixture, 覆盖 scan+verdict 全分支。"""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_import as mi


def wstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(path, ttypes, extra_arch=True):
    """构造最小 GGUF: version=3, 若干 metadata KV (含一个 array), 若干 tensor 条目。
    ttypes: [(name, ndims, shape, type_id), ...]"""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQI", 3, len(ttypes), 2 + (1 if extra_arch else 0)))
        # KV1: general.architecture (string)
        f.write(wstr("general.architecture"))
        f.write(struct.pack("<I", 8))
        f.write(wstr("qwen3"))
        # KV2: tokenizer.ggml.model (string)
        f.write(wstr("tokenizer.ggml.model"))
        f.write(struct.pack("<I", 8))
        f.write(wstr("gpt2"))
        # KV3 (array of uint32) — 测数组跳过
        if extra_arch:
            f.write(wstr("general.file_type"))
            f.write(struct.pack("<I", 9))          # array type
            f.write(struct.pack("<I", 4))          # elem uint32
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<II", 1, 2))
        for name, ndims, shape, tt in ttypes:
            f.write(wstr(name))
            f.write(struct.pack("<I", ndims))
            for d in shape:
                f.write(struct.pack("<q", d))
            f.write(struct.pack("<I", tt))
            f.write(struct.pack("<q", 0))
        f.write(b"\x00" * 64)  # 伪数据区


def main():
    d = tempfile.mkdtemp(prefix="ninfer_import_test_")
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        print(("%s  %s%s" % ("PASS" if cond else "FAIL", label, extra)))
        if not cond:
            ok = False

    # 1. ninfer 真实 artifact (引擎白名单身份)
    art = r"C:\Users\User\Documents\ziqinzhang\models\Qwen3.8-27B-Huihui-Abliterated-NInfer-DFlash2\qwen3_8_27b_nvfp4_dflash2.ninfer"
    if os.path.exists(art):
        s = mi.scan_path(art)
        check("ninfer kind", s.kind == "ninfer", " | model_id=%s weights=%s" % (s.model_id, s.weights_id))
        check("ninfer id", s.model_id == "qwen3.8-27b")
        v = mi.verdict(s, mi.GpuInfo(present=True, name="RTX 5090D", vram_total_gb=32.0, vram_free_gb=30.0))
        check("ninfer verdict run", v.action == "run", " | %s" % v.title)
        v2 = mi.verdict(s, mi.GpuInfo(present=True, name="GTX 1060", vram_total_gb=6.0, vram_free_gb=5.0))
        check("ninfer vram short", v2.action == "vram_short", " | %s" % v2.title)
    else:
        print("skip ninfer artifact (missing)")

    # 2. HF fixture: 几何匹配 (5120/64) -> convert_groupwise
    hf = os.path.join(d, "hf_ok")
    os.makedirs(hf)
    with open(os.path.join(hf, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"architectures":["Qwen3_5ForConditionalGeneration"],"model_type":"qwen3_5",'
                '"text_config":{"hidden_size":5120,"num_hidden_layers":64}}')
    with open(os.path.join(hf, "model-00001-of-00001.safetensors"), "wb") as f:
        f.write(b"\x00" * (2 * (27 << 30) // 2) )  # 27B bf16 假体积 — 太大, 用 2GB
    os.remove(os.path.join(hf, "model-00001-of-00001.safetensors"))
    with open(os.path.join(hf, "model-00001-of-00001.safetensors"), "wb") as f:
        f.write(b"\x00" * (2 << 30))
    s = mi.scan_path(hf)
    check("hf kind", s.kind == "hf", " | %s" % s.arch_note)
    check("hf family exact", s.model_id == "qwen3.8-27b", " | %s" % s.model_id)
    v = mi.verdict(s, mi.GpuInfo(present=True, name="RTX 5090D", vram_total_gb=32.0, vram_free_gb=30.0))
    check("hf verdict convert", v.action == "convert_groupwise", " | %s" % v.title)
    v2 = mi.verdict(s, mi.GpuInfo(present=True, name="GTX 1060", vram_total_gb=4.0, vram_free_gb=3.0))
    check("hf vram short", v2.action == "vram_short", " | %s" % v2.title)

    # 3. HF fixture 几何不匹配 (5120/32) -> unsupported_arch
    hf2 = os.path.join(d, "hf_32l")
    os.makedirs(hf2)
    with open(os.path.join(hf2, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"model_type":"qwen3_5","text_config":{"hidden_size":5120,"num_hidden_layers":32}}')
    with open(os.path.join(hf2, "m.safetensors"), "wb") as f:
        f.write(b"\x00" * (1 << 20))
    s = mi.scan_path(hf2)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("hf 32l unsupported", v.action == "unsupported_arch", " | %s" % v.title)

    # 4. GGUF F16 (白名单文件名) -> convert_groupwise
    g1 = os.path.join(d, "qwen3.8-27b-f16.gguf")
    build_gguf(g1, [("token_embd.weight", 2, [5120, 151936], 1),
                    ("blk.0.attn_q.weight", 2, [5120, 5120], 1)])
    s = mi.scan_path(g1)
    check("gguf kind", s.kind == "gguf", " | %s" % s.quant_name)
    check("gguf no kquant issue", not any("K" in i for i in s.issues), " | %s" % s.issues)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("gguf f16 convert", v.action == "convert_groupwise", " | %s" % v.title)

    # 5. GGUF k-quant -> unsupported_quant
    g2 = os.path.join(d, "qwen3.8-27b-Q4_K_M.gguf")
    build_gguf(g2, [("token_embd.weight", 2, [5120, 151936], 11)])
    s = mi.scan_path(g2)
    check("gguf kquant detect", any("K" in i for i in s.issues), " | %s" % s.quant_name)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("gguf kquant rejected", v.action == "unsupported_quant", " | %s" % v.title)

    # 6. 未知路径
    s = mi.scan_path(os.path.join(d, "nothing_here"))
    check("unknown kind", s.kind == "unknown")
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("unknown verdict", v.action == "unknown")

    # 7. 环境自检
    env = mi.env_report()
    check("env report items", len(env["items"]) >= 3)

    # 8. 真实探针 config (几何匹配的官方多模态版)
    probe = r"C:\Users\User\Documents\ziqinzhang\data\qwen38_config_probe.json"
    if os.path.exists(probe):
        pdir = os.path.join(d, "probe")
        os.makedirs(pdir)
        with open(os.path.join(pdir, "config.json"), "w", encoding="utf-8") as f:
            f.write(open(probe, encoding="utf-8").read())
        with open(os.path.join(pdir, "m.safetensors"), "wb") as f:
            f.write(b"\x00" * (1 << 20))
        s = mi.scan_path(pdir)
        v = mi.verdict(s, mi.GpuInfo(present=True, vram_total_gb=32.0))
        check("real probe config family", s.model_id == "qwen3.8-27b",
              " | %s | verdict=%s" % (s.arch_note, v.action))
    print("== %s" % ("ALL PASS" if ok else "HAS FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
