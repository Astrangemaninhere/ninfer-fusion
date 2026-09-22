# -*- coding: utf-8 -*-
"""model_import 自测: 构造最小合法 GGUF/HF fixture, 覆盖 scan+verdict 全分支。"""
import atexit
import os
import shutil
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_import as mi


def wstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(path, ttypes, extra_arch=True, arch="qwen3", uint32_kvs=None):
    """构造最小合法 GGUF: version=3, 若干 metadata KV (含一个 array), 若干 tensor 条目。
    ttypes: [(name, ndims, shape, type_id), ...]
    uint32_kvs: {key: int} — 额外的 uint32 KV (ktype 4)。**必须能按规范写**, 因为
    层数记账 (qwen35.block_count / qwen35.nextn_predict_layers) 就在这两个键里, 而在
    真实文件上它们正是 ktype 4 (实测 Ornith-1.5-9B-Q4_K_M.gguf, 42 个 KV)。

    头部必须写满 gguf 规范要求的 24 字节:
        magic(4) + version(u32) + tensor_count(u64) + metadata_kv_count(u64)
    这里原来写的是 struct.pack("<IQI", ...) (16 字节), 与解析器犯的是同一个错误 ——
    两边一起错, 所以这个 fixture 永远测不出解析器读错偏移。真实文件
    (Ornith-1.5-9B-Q4_K_M.gguf) 的头部是 24 字节, 旧解析器只消费 20 字节, 于是把第一
    个 KV 的名字长度读成 85_899_345_920 (80 GiB) 并抛出空消息的 MemoryError。
    对照实现: tools/archkit/gguf_tensors.py:51 用的是 <IQQ。
    """
    uint32_kvs = uint32_kvs or {}
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(ttypes),
                            2 + (1 if extra_arch else 0) + len(uint32_kvs)))
        # KV1: general.architecture (string)
        f.write(wstr("general.architecture"))
        f.write(struct.pack("<I", 8))
        f.write(wstr(arch))
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
        for key, value in uint32_kvs.items():
            f.write(wstr(key))
            f.write(struct.pack("<I", 4))          # uint32, as the real file has it
            f.write(struct.pack("<I", int(value)))
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
    # 每条退出路径都清掉 (含异常): 这个自测曾经**每次都**留下一个多 GB 的 tempdir,
    # 而 /tmp 是 11 GiB 的 tmpfs —— 跑够几次它就把 /tmp 填满, 之后与本测试无关的
    # 任何进程都会开始 ENOSPC。实测: 连续 3 次后 /tmp 到 57%。
    atexit.register(shutil.rmtree, d, ignore_errors=True)
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
        # 只要"文件够大"这一个效果, 2 GiB 就够(fixture 只用 stat().st_size)。
        # 这里原来写的是 b"\x00" * (2 * (27 << 30) // 2), 注释却写"太大, 用 2GB":
        #   2 * (27 << 30) // 2 == 27 << 30 == 28_991_029_248 字节 == 27 GiB
        # "改成 2GB"只写进了注释, 从未落到表达式里。后果有两层:
        #   1) 先在内存里真的构造一个 27 GiB 的 bytes 对象(本机 21 GiB RAM, 靠 50 GiB
        #      swap 才可能成功 —— 就是那次把整机拖到几乎杀掉一个 30 分钟构建的元凶);
        #   2) 写进 TMPDIR, 而 /tmp 是 11 GiB 的 tmpfs: 写到 ~10.7 GB 时 ENOSPC,
        #      紧接着的 os.remove() 永远执行不到, 留下一份和 /tmp/ninfer_import_test_*
        #      同名的 10.7 GB 残留文件。
        # 上一轮把它降到 2 GiB, 但 2 GiB 仍然真的写盘, 而这个测试**每跑一次**都建一个
        # tempdir 且从不清: 实测跑 3 次后 /tmp(11 GiB tmpfs) 到 57%, 跑到第 6 次就满,
        # 之后**与本测试无关的任何东西**都会开始 ENOSPC (实测: 本轮我先撞了两次
        # "OSError: [Errno 28] No space left on device", 与源码改动无关)。
        # 现在用稀疏文件: truncate 只改 st_size, 不分配数据块, 所以 tmpfs 占用 ~0,
        # 语义完全不变 (这个 fixture 唯一的用途就是 stat().st_size 够大)。
        f.truncate(2 << 30)
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

    # 4. GGUF F16 (白名单文件名 + 与支持规格一致的几何) -> convert_groupwise
    #    几何是必需的: 家族不再只看文件名, token_embd 给宽度、blk.N 给层数,
    #    都对上 (5120 / 64) 才承诺"能转"。
    g1 = os.path.join(d, "qwen3.8-27b-f16.gguf")
    build_gguf(g1, [("token_embd.weight", 2, [5120, 151936], 1)] +
               [("blk.%d.attn_q.weight" % i, 2, [5120, 5120], 1) for i in range(64)])
    s = mi.scan_path(g1)
    check("gguf kind", s.kind == "gguf", " | %s" % s.quant_name)
    check("gguf geometry read", (s.hidden_size, s.n_layers) == (5120, 64),
          " | hidden=%s layers=%s" % (s.hidden_size, s.n_layers))
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
    check("kquant refusal names the undecodable type and the decodable set",
          "Q3_K" in v.detail and "Q4_K" in v.detail and "Q6_K" in v.detail,
          " | %s" % v.detail[:120])
    check("kquant refusal points at the engine-side gap too",
          "target" in v.detail, " | %s" % v.detail[:120])

    # 5a2. Q4_K/Q6_K 现在有反量化器 -> 不许再报"K 系列不支持"
    #      (旧的 {F32,F16,BF16} 字面量在这里会误报; 现在类型集是问读者要的)
    g2b = os.path.join(d, "qwen3.8-27b-Q4_K_M-decodable.gguf")
    build_gguf(g2b, [("token_embd.weight", 2, [5120, 151936], 12)] +
               [("blk.%d.attn_q.weight" % i, 2, [5120, 5120], 14) for i in range(64)])
    s = mi.scan_path(g2b)
    check("q4_k/q6_k no longer reported as unsupported",
          not any("K" in i for i in s.issues), " | issues=%s" % s.issues)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("q4_k/q6_k with matching geometry -> convert",
          v.action == "convert_groupwise", " | %s" % v.title)

    # 5a3. Q3_K 仍然解不了 -> 必须拒绝, 而且要指名是哪一个类型
    g2c = os.path.join(d, "qwen3.8-27b-Q3_K.gguf")
    build_gguf(g2c, [("token_embd.weight", 2, [5120, 151936], 11)] +
               [("blk.%d.attn_q.weight" % i, 2, [5120, 5120], 11) for i in range(64)])
    s = mi.scan_path(g2c)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("q3_k still refused", v.action == "unsupported_quant", " | %s" % v.title)

    # 5b. 名字像支持的规格, 几何不是 -> 不许承诺转换 (旧代码只认文件名)
    g3 = os.path.join(d, "qwen3.8-27b-f16-wrongshape.gguf")
    build_gguf(g3, [("token_embd.weight", 2, [4096, 248320], 1)] +
               [("blk.%d.attn_q.weight" % i, 2, [4096, 4096], 1) for i in range(33)])
    s = mi.scan_path(g3)
    check("gguf geometry mismatch read", (s.hidden_size, s.n_layers) == (4096, 33),
          " | hidden=%s layers=%s" % (s.hidden_size, s.n_layers))
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("gguf wrong geometry not promised", v.action == "unsupported_arch",
          " | %s" % v.title)

    # 5c. 架构来自 GGUF 自己的元数据, 不是文件名
    g4 = os.path.join(d, "some-random-download-name.gguf")
    build_gguf(g4, [("token_embd.weight", 2, [5120, 151936], 1)] +
               [("blk.%d.attn_q.weight" % i, 2, [5120, 5120], 1) for i in range(64)],
               arch="qwen3")
    s = mi.scan_path(g4)
    check("gguf arch from metadata (renamed file)",
          s.model_id == "qwen3.8-27b", " | model_id=%r arch_note=%s" % (s.model_id, s.arch_note))
    # 反过来: 文件名说是 qwen3.8-27b, 但元数据说是别的架构 -> 不许认
    g5 = os.path.join(d, "qwen3.8-27b-f16-mismatched-arch.gguf")
    build_gguf(g5, [("token_embd.weight", 2, [4096, 248320], 1)] +
               [("blk.%d.ssm_out.weight" % i, 2, [4096, 4096], 1) for i in range(33)],
               arch="ornith")
    s = mi.scan_path(g5)
    check("gguf metadata beats file name", s.model_id != "qwen3.8-27b",
          " | model_id=%r arch_note=%s" % (s.model_id, s.arch_note))
    check("gguf unmapped tensors named", s.unmapped and "ssm_out" in s.unmapped[0],
          " | %s" % s.unmapped)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("unmapped gguf refusal lists the gap",
          v.action == "unsupported_arch" and "ssm_out" in " ".join(v.tips),
          " | tips=%s" % v.tips)

    # 5d. 截断的 GGUF -> 必须说"读不出来(文件截断)", 不能说"家族不支持"
    g6 = os.path.join(d, "truncated.gguf")
    with open(g6, "wb") as f:
        f.write(b"GGUF" + b"\x03\x00\x00\x00")      # 8 字节, 头部要 24
    s = mi.scan_path(g6)
    check("truncated gguf unreadable", s.unreadable, " | issues=%s" % s.issues)
    check("truncated gguf names the size", any("24" in i for i in s.issues),
          " | issues=%s" % s.issues)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("truncated gguf verdict", v.action == "unreadable", " | %s" % v.title)
    check("truncated gguf not blamed on family",
          "家族不在支持列表" not in v.detail, " | %s" % v.detail[:100])

    # 5e. 真实文件 (存在才跑): 旧解析器在这个文件上是空消息的 MemoryError
    real_gguf = (r"/mnt/c/Users/User/Documents/ziqinzhang/models/ornith-1.5-9b/"
                 r"Ornith-1.5-9B-Q4_K_M.gguf")
    if os.path.exists(real_gguf):
        s = mi.scan_path(real_gguf)
        v = mi.verdict(s, mi.GpuInfo(present=True))
        # 层数是 **32**, 不是 33。实测这个文件: 442 张量, 33 个块 (blk.0..blk.32),
        # qwen35.block_count=33, qwen35.nextn_predict_layers=1 (两个键都是 ktype 4),
        # 而且 blk.32 同时带着完整 attn_*/ffn_* **和** blk.32.nextn.* —— 所以它是草稿
        # 块, 不是第 33 层解码层。33 是"块数", max(blk.N)+1 数的就是块数。
        # 改前这里钉的是 33, 而 tools/archkit/gguf_spec.py 对同一个文件写的是
        # geometry.layers=32 + mtp=1, 于是向导指着一个 spec 说"33 层"。
        check("real gguf parses (442 tensors)", s.kind == "gguf" and not s.unreadable
              and s.n_layers == 32 and s.hidden_size == 4096,
              " | hidden=%s layers=%s quant=%s issues=%s"
              % (s.hidden_size, s.n_layers, s.quant_name, s.issues))
        # 同一个文件、两个读者, 必须给出同一个深度 —— 这条断言就是"向导与 spec 一致"
        # 本身, 不是它的复述: build_spec 走的是 gguf_spec.py 自己的元数据路径。
        try:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from tools.archkit.gguf_spec import build_spec
            spec = build_spec(real_gguf)
            check("wizard depth == gguf_spec depth for the same file",
                  s.n_layers == spec["geometry"]["layers"] == 32
                  and spec.get("mtp", {}).get("mtp_num_hidden_layers") == 1,
                  " | wizard=%s spec=%s mtp=%s"
                  % (s.n_layers, spec["geometry"]["layers"], spec.get("mtp")))
        except Exception as exc:                        # noqa: BLE001
            check("wizard depth == gguf_spec depth for the same file", False,
                  " | cross-check could not run: %r" % (exc,))
        # The bytes are readable (Q4_K/Q6_K have decoders, 442/442 names mapped),
        # so the refusal is about the engine-side target, not about the format.
        check("real gguf not refused for quantisation",
              v.action != "unsupported_quant", " | action=%s %s" % (v.action, v.title))
        check("real gguf names fully mapped", not s.unmapped, " | %s" % s.unmapped)
        check("real gguf verdict is the engine-side gap",
              v.action == "unsupported_arch", " | action=%s" % v.action)
        # 类型名必须与真实 ggml 表一致: Ornith 是 Q4_K x223 / Q6_K x35 / F32 x184
        # (tools/archkit/specs/ornith-1.5-9b-q4-k-m_spec.json 的 quant_formats)
        check("real gguf quant names match the spec",
              "Q4_K" in s.quant_name and "Q6_K" in s.quant_name
              and "F32" in s.quant_name and "Q5_K" not in s.quant_name,
              " | %s" % s.quant_name)
    else:
        print("skip real gguf (missing)")

    # 5g. 层数记账 (合成, 不依赖仓库外的材料): block_count 含 nextn/草稿块。
    #     形状照抄真文件的实测结果 —— 33 个块, 每块有 attn_q, **并且** blk.32 另有
    #     blk.32.nextn.eh_proj (草稿块的 attn/ffn 张量也在 blk.32 里, 所以"跳过只带
    #     nextn 张量的块"这条也错; 只有元数据能点名草稿块)。改前报 33, 改后报 32。
    _draft_tensors = ([("token_embd.weight", 2, [4096, 248320], 1)]
                      + [("blk.%d.attn_q.weight" % i, 2, [4096, 4096], 1)
                         for i in range(33)]
                      + [("blk.32.nextn.eh_proj.weight", 2, [8192, 4096], 1)])
    g8 = os.path.join(d, "ornith-shaped-nextn.gguf")
    build_gguf(g8, _draft_tensors, arch="qwen35",
               uint32_kvs={"qwen35.block_count": 33, "qwen35.nextn_predict_layers": 1})
    s = mi.scan_path(g8)
    check("nextn block is not counted as a decoder layer",
          (s.hidden_size, s.n_layers) == (4096, 32),
          " | hidden=%s layers=%s issues=%s" % (s.hidden_size, s.n_layers, s.issues))
    check("consistent nextn accounting raises no issue",
          not any("nextn" in i or "layer accounting" in i for i in s.issues),
          " | issues=%s" % s.issues)
    # 负对照: 同一批张量, 去掉那两个 KV —— 规则一去, 结论就变回 33 (改前的数字),
    # 并且不一致必须被点名, 而不是悄悄按其中一个数记账。
    g9 = os.path.join(d, "ornith-shaped-nextn-no-kv.gguf")
    build_gguf(g9, _draft_tensors, arch="qwen35")
    s = mi.scan_path(g9)
    check("control: without the metadata the same tensors still read as 33",
          s.n_layers == 33, " | layers=%s" % s.n_layers)
    check("control: and the disagreement is named, not silently resolved",
          any("nextn" in i for i in s.issues), " | issues=%s" % s.issues)
    # 再一个负对照: 33 个块但**没有** .nextn.* 张量 (5b 那种普通 33 层模型) 仍然是 33,
    # 说明这条规则不是"把 33 一律减 1"。
    g10 = os.path.join(d, "plain-33-block.gguf")
    build_gguf(g10, [("token_embd.weight", 2, [4096, 248320], 1)]
               + [("blk.%d.attn_q.weight" % i, 2, [4096, 4096], 1) for i in range(33)],
               arch="qwen35")
    s = mi.scan_path(g10)
    check("control: 33 ordinary blocks stay 33 (the rule is not 'minus one')",
          s.n_layers == 33, " | layers=%s" % s.n_layers)

    # 5f. ggml id 2 是 Q4_0, 旧表把它当 BF16 -> 会被当成"半精度可转"
    g7 = os.path.join(d, "qwen3.8-27b-Q4_0.gguf")
    build_gguf(g7, [("token_embd.weight", 2, [5120, 151936], 2)] +
               [("blk.%d.attn_q.weight" % i, 2, [5120, 5120], 2) for i in range(64)])
    s = mi.scan_path(g7)
    check("q4_0 not labelled bf16", "Q4_0" in s.quant_name and "BF16" not in s.quant_name,
          " | %s" % s.quant_name)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("q4_0 refused", v.action == "unsupported_quant", " | %s" % v.title)

    # 6. 未知路径
    s = mi.scan_path(os.path.join(d, "nothing_here"))
    check("unknown kind", s.kind == "unknown")
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("unknown verdict", v.action == "unknown")

    # 7. 环境自检
    env = mi.env_report()
    check("env report items", len(env["items"]) >= 3)

    # 7b. config.json 坏掉 -> 必须说"config 读不出来", 不能甩锅给"架构不支持"
    hj = os.path.join(d, "hf_bad_config")
    os.makedirs(hj)
    with open(os.path.join(hj, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"architectures": ["Qwen3ForCausalLM", ')          # 故意截断
    with open(os.path.join(hj, "m.safetensors"), "wb") as f:
        f.write(b"\x00" * (1 << 20))
    s = mi.scan_path(hj)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("bad config verdict", v.action == "unreadable", " | %s" % v.title)
    check("bad config not blamed on arch",
          "架构" not in v.title and "JSON" in v.detail, " | %s" % v.detail[:90])
    check("bad config issue recorded", bool(s.config_error), " | %s" % s.config_error)

    # 7c. 有 config.json 没有权重 -> 必须说缺权重, 不能又说"没有模型文件"
    hw = os.path.join(d, "hf_no_weights")
    os.makedirs(hw)
    with open(os.path.join(hw, "config.json"), "w", encoding="utf-8") as f:
        f.write('{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}')
    s = mi.scan_path(hw)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("no-weights verdict", v.action == "unreadable", " | %s" % v.title)
    check("no-weights names weights",
          "safetensors" in v.detail, " | %s" % v.detail[:90])

    # 7d. 空目录仍然走"没有模型文件"这一条
    he = os.path.join(d, "empty_dir")
    os.makedirs(he)
    s = mi.scan_path(he)
    v = mi.verdict(s, mi.GpuInfo(present=True))
    check("empty dir verdict", v.action == "unknown", " | %s" % v.title)

    # 7e. 缺口报告不许把"没有 gaps 字段"当成"没有缺口"
    import json as _json
    fake_out = os.path.join(d, "archkit_out")
    os.makedirs(os.path.join(fake_out, "m1"))
    with open(os.path.join(fake_out, "m1", "manifest.json"), "w", encoding="utf-8") as f:
        _json.dump({"model_id": "m1", "gaps": [
            {"need": "attention:sliding_window(1024)", "tier": "new_op", "action": "..."}]}, f)
    with open(os.path.join(fake_out, "m1", "arch_manifest.json"), "w", encoding="utf-8") as f:
        _json.dump({"model_id": "m1", "geometry": {"hidden": 5376}, "layer_types": ["swa"],
                    "gui_hint": {"supported": True, "import_note": "支持该家族转换"}}, f)
    os.makedirs(os.path.join(fake_out, "m2"))
    with open(os.path.join(fake_out, "m2", "arch_manifest.json"), "w", encoding="utf-8") as f:
        _json.dump({"model_id": "m2", "geometry": {"hidden": 5376},
                    "gui_hint": {"supported": True}}, f)
    # 7f. 四种"gaps 的缺席"必须走同一个判据. m3 = 显式为 null (item3 写者写的那种),
    #     m4 = 非列表, m5 = 带 gaps 列表但写者自己声明 gaps_measured:false。
    #     没有这几个 fixture, "缺失 vs null" 这类分歧就只能在运行时靠运气被发现 ——
    #     m2/m3 的值不同、结论必须相同, 这正是要钉住的那条。
    for mid, body in (
            # m3 = 显式 null 且**不写** gaps_measured —— 纯粹是 item3 写者的半个形状,
            # 单看 null 这一项。m5 = 带列表但写者自己否认, 是 item3 写者的完整形状。
            ("m3", {"model_id": "m3", "gaps": None,
                    "gui_hint": {"engine_supported": None,
                                 "import_note": "该家族可被本生成器翻译"}}),
            ("m4", {"model_id": "m4", "gaps": "none"}),
            ("m5", {"model_id": "m5", "gaps": [], "gaps_measured": False}),
            # m7 = item3 写者整份形状的逐字复刻 (gaps: null + gaps_measured: false).
            # 这一条是本题的原始攻击载荷, 单独钉住它的 shape 标签 (flag_false:null),
            # 因为曾经被描述成"带了列表但说 false" —— 对一份 gaps 为 null 的清单是错的。
            ("m7", {"model_id": "m7", "gaps": None, "gaps_measured": False,
                    "gaps_source": "none: gen_target.py v1 has no operator catalogue",
                    "verified_against_engine": False,
                    "gui_hint": {"supported": True,
                                 "import_note": "支持该家族转换"}}),
            # m6 是对照组: 空列表且不否认 ⇒ 这是一次真的测量, 必须判 clear。
            ("m6", {"model_id": "m6", "gaps": []}),
            # m8 = gaps 是列表, 但元素不是对象。这一条以前**不是判错而是崩**:
            #   tools/gui/model_import.py:1252 `g.get("need")` 对字符串抛 AttributeError,
            #   异常穿过 import_gap_report() 和 main() 冒到调用方 ⇒ GUI 拿到的是 Python
            #   栈, 退出码 1 —— 与"没有清单"同一个码, 所以调用方连"读不到文件"和
            #   "文件读到了但内容是垃圾"都分不开。现在它必须落到 UNKNOWN, 且 rc=2。
            ("m8", {"model_id": "m8", "gaps": [
                "attention:linear(gdn)",
                {"need": "head:tied=true", "tier": "new_op", "action": "x"}]}),
            # m9/m10 = 写者把旗标拼成 `measured`。本树**没有任何地方读这个键**
            #   (唯一有效拼法是 gaps_measured), 于是写者的声明被无声忽略:
            #   m10 是完整的假绿灯载荷 —— 写者明确说"这份清单没测过", 而旧读者因为
            #   看到 gaps 是个空列表, 判成 CLEAR / rc=0 / "引擎可以直接吃"。
            ("m9", {"model_id": "m9", "gaps": [], "measured": True}),
            ("m10", {"model_id": "m10", "gaps": [], "measured": False}),
            # m11/m12 = gaps 是列表、元素也是对象, 但**层型读不出来** —— 键根本没有
            #   (m11), 或者拼成别的词 (m12)。这两条是 2026-09-14 用真 CLI 打出来的
            #   假绿灯: 报告同时打印 `gaps: 1 (hook 0 / new_op 0 / post 0)`、
            #   `VERDICT: CLEAR — config.h generated, no new_op gaps` 和一整行
            #   `new_op:attention:linear(gdn)`, rc=0 (加 --require-servable 也是 0)。
            #   成因: 结论只由 `tier == "new_op"` 决定, 所以"层型读不出"就被当成
            #   "不算拦截"。判据与 bad_element 同类 —— 读不出的行 = 不是测量。
            ("m11", {"model_id": "m11", "gaps": [
                {"need": "new_op:attention:linear(gdn)", "action": "x"}]}),
            ("m12", {"model_id": "m12", "gaps": [
                {"need": "new_op:attention:linear(gdn)", "tier": "new-op",
                 "action": "x"}]}),
    ):
        os.makedirs(os.path.join(fake_out, mid))
        with open(os.path.join(fake_out, mid, "manifest.json"), "w", encoding="utf-8") as f:
            _json.dump(body, f)
    # m6 的清单目录里**真的**放一个 config.h, m1 的目录里什么都不放。header_file 是
    # 关于目录的事实, 所以这两条是同一个判据的两侧, 而不是"blocked 就等于某个文件名"。
    with open(os.path.join(fake_out, "m6", "config.h"), "w", encoding="utf-8") as f:
        f.write("// fixture header\n")
    saved_out = mi.ARCHKIT_OUT
    mi.ARCHKIT_OUT = __import__("pathlib").Path(fake_out)
    try:
        rep = mi.import_gap_report("m1")
        check("gap report finds manifest.json", rep["ok"] and rep["blocked"],
              " | blocked=%s blocking=%s" % (rep["blocked"], rep["blocking"]))
        rep2 = mi.import_gap_report("m2")
        check("arch_manifest.json is found too", rep2.get("manifest", "").endswith(
            "arch_manifest.json"), " | %s" % rep2.get("manifest"))
        # `ok` means "found and parsed" (see the docstring); the load-bearing field is
        # `gaps_measured` + `verdict`. Asserting ok is False here would pass on a
        # report that cannot be told apart from "the file is missing", and would make
        # the rc=2 branch in main() unreachable -- the shape this test was written to
        # pin has to be asserted with the fields that actually carry it.
        check("no-gaps manifest is NOT an all-clear",
              rep2["ok"] is True and rep2["gaps_measured"] is False
              and rep2["verdict"] == "unmeasured" and rep2.get("error"),
              " | ok=%s gaps_measured=%s verdict=%s error=%s"
              % (rep2["ok"], rep2["gaps_measured"], rep2["verdict"],
                 rep2.get("error", "")[:70]))
        check("no-gaps manifest says what shape was found",
              rep2.get("gap_shape") == "missing", " | shape=%s" % rep2.get("gap_shape"))
        null_rep = mi.import_gap_report("m3")
        check("gaps:null is NOT an all-clear either",
              null_rep["gaps_measured"] is False and null_rep["verdict"] == "unmeasured"
              and null_rep["ok"] is True,
              " | ok=%s gaps_measured=%s verdict=%s shape=%s"
              % (null_rep["ok"], null_rep["gaps_measured"], null_rep["verdict"],
                 null_rep.get("gap_shape")))
        check("gaps:null is told apart from a missing field",
              null_rep.get("gap_shape") == "null", " | shape=%s" % null_rep.get("gap_shape"))
        check("missing key and explicit null reach the SAME verdict",
              rep2["verdict"] == null_rep["verdict"] == "unmeasured"
              and rep2["gaps_measured"] is False and null_rep["gaps_measured"] is False
              and rep2["header_file"] == null_rep["header_file"] == "unknown",
              " | m2=%s/%s m3=%s/%s"
              % (rep2["verdict"], rep2["header_file"], null_rep["verdict"],
                 null_rep["header_file"]))
        for mid, shape in (("m4", "not_a_list:str"), ("m5", "flag_false:list"),
                           ("m7", "flag_false:null"), ("m8", "bad_element:str@0"),
                           ("m9", "alias_key:measured"), ("m10", "alias_key:measured"),
                           ("m11", "bad_tier:missing@0"), ("m12", "bad_tier:new-op@0")):
            r = mi.import_gap_report(mid)
            check("%s is NOT an all-clear" % mid,
                  r["verdict"] == "unmeasured" and r["gaps_measured"] is False,
                  " | verdict=%s shape=%s" % (r["verdict"], r.get("gap_shape")))
            check("%s shape named" % mid, r.get("gap_shape") == shape,
                  " | shape=%s" % r.get("gap_shape"))
        # 负对照 (seam 4): 去掉上面那条"层型必须在 TIER_ORDER 里"的规则, m11/m12 会
        # 立刻变回 CLEAR —— 把**改前的判据**原样实现, 断言它把这两份清单当成了测量。
        def _pre_seam4_predicate(man):
            """改前的 manifest_gap_state: 只要求元素是对象, 不读它的 tier。"""
            if man.get("gaps_measured") is False or "measured" in man:
                return False
            raw = man.get("gaps")
            if not isinstance(raw, list):
                return False
            return all(isinstance(g, dict) for g in raw)

        def _pre_seam4_verdict(man):
            """改前的 CLEAR/BLOCKED 判定: 只有 tier 恰好等于 new_op 才拦截。"""
            if not _pre_seam4_predicate(man):
                return "unmeasured"
            return "blocked" if any(g.get("tier") == "new_op"
                                    for g in man["gaps"]) else "clear"
        m11_body = {"gaps": [{"need": "new_op:attention:linear(gdn)", "action": "x"}]}
        m12_body = {"gaps": [{"need": "new_op:attention:linear(gdn)",
                              "tier": "new-op", "action": "x"}]}
        check("control: the PRE-seam4 predicate called m11/m12 measurements",
              _pre_seam4_predicate(m11_body) is True
              and _pre_seam4_predicate(m12_body) is True,
              " | pre=%s/%s" % (_pre_seam4_predicate(m11_body),
                                _pre_seam4_predicate(m12_body)))
        check("control: PRE-seam4 rendered both as CLEAR (the false all-clear)",
              _pre_seam4_verdict(m11_body) == "clear"
              and _pre_seam4_verdict(m12_body) == "clear",
              " | pre=%s/%s" % (_pre_seam4_verdict(m11_body),
                                _pre_seam4_verdict(m12_body)))
        check("seam4: control still clears an empty list (the rule is about the row)",
              _pre_seam4_verdict({"gaps": []}) == "clear"
              and mi.import_gap_report("m6")["verdict"] == "clear",
              " | pre=%s now=%s" % (_pre_seam4_verdict({"gaps": []}),
                                    mi.import_gap_report("m6")["verdict"]))
        check("seam4: a row that DOES carry tier=new_op still blocks",
              _pre_seam4_verdict({"gaps": [{"tier": "new_op"}]}) == "blocked"
              and mi.import_gap_report("m1")["verdict"] == "blocked",
              " | pre=%s now=%s" % (_pre_seam4_verdict({"gaps": [{"tier": "new_op"}]}),
                                    mi.import_gap_report("m1")["verdict"]))
        # 层型词汇表是怎么定的: adapt.py 只会写这四个, 盘上每一份带 gaps 的清单也只用
        # 这四个 —— 所以"读不出层型"确实不是一个正常写者会产生的形状, 而是漂移信号。
        check("the tier vocabulary is adapt.py's four tiers",
              tuple(mi.TIER_ORDER) == ("hook", "new_op", "covered", "post"),
              " | %s" % (mi.TIER_ORDER,))
        # 负对照 (seam 2): 去掉"含 measured 键就拒绝"这条规则, m10 会变回 CLEAR。
        # 这里把**改前的判据**原样实现一遍, 断言它与现在的判据不同 —— 否则这两个
        # fixture 就只是"又一个通过了"的用例, 证明不了规则在起作用。
        def _pre_seam2_predicate(man):
            """改前的 manifest_gap_state: 只看 gaps_measured, 不看 measured。"""
            if man.get("gaps_measured") is False:
                return False
            if "gaps" not in man:
                return False
            raw = man.get("gaps")
            return isinstance(raw, list)
        pre = {}
        for mid, body in (("m9", {"gaps": [], "measured": True}),
                          ("m10", {"gaps": [], "measured": False}),
                          ("m6", {"gaps": []})):
            pre[mid] = _pre_seam2_predicate(body)
        check("control: the PRE-seam2 predicate called m10 a measurement",
              pre["m10"] is True and pre["m9"] is True and pre["m6"] is True,
              " | pre=%s" % pre)
        check("seam2: those two now reach UNMEASURED while the true control stays CLEAR",
              mi.import_gap_report("m10")["gaps_measured"] is False
              and mi.import_gap_report("m9")["gaps_measured"] is False
              and mi.import_gap_report("m6")["gaps_measured"] is True,
              " | m10=%s m9=%s m6=%s" % (mi.import_gap_report("m10")["gaps_measured"],
                                         mi.import_gap_report("m9")["gaps_measured"],
                                         mi.import_gap_report("m6")["gaps_measured"]))
        # 负对照 (seam 1): 同样的形状在改前是 AttributeError, 不是判错。
        def _pre_seam1_would_crash(man):
            try:
                for g in (man.get("gaps") or []):
                    g.get("need")
            except AttributeError:
                return True
            return False
        check("control: the PRE-seam1 row loop raises on m8's list",
              _pre_seam1_would_crash({"gaps": ["attention:linear(gdn)"]}) is True,
              " | pre=AttributeError")
        check("seam1: import_gap_report survives it and names the element",
              mi.import_gap_report("m8")["gap_shape"] == "bad_element:str@0",
              " | %s" % mi.import_gap_report("m8")["gap_shape"])
        check("item3's writer shape reaches UNMEASURED, not CLEAR",
              mi.import_gap_report("m7")["verdict"] == "unmeasured"
              and mi.import_gap_report("m7")["header_file"] == "unknown",
              " | %s" % mi.import_gap_report("m7")["verdict"])
        # 负对照: 去掉"空列表算一次测量"这条, m6 就会和 m2..m5 一样变成 unmeasured。
        r6 = mi.import_gap_report("m6")
        check("control: an empty gaps list DOES read as clear",
              r6["gaps_measured"] is True and r6["verdict"] == "clear" and not r6["blocked"],
              " | verdict=%s gaps_measured=%s" % (r6["verdict"], r6["gaps_measured"]))
        # ---- seam 3: 退出码就是结论 ----------------------------------------
        # 改前 BLOCKED 和 CLEAR 都是 0: "结论说不能跑, 退出码说没问题"。这里从
        # **真的 CLI 入口** (main()) 取码, 不是从辅助函数取 —— 契约在 main() 上。
        # 报告正文被吞掉: 这几行测的是退出码, 把 9 份报告打进日志只会淹没断言。
        import contextlib
        import io as _io

        def _rc_cli(*argv):
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                return mi.main(list(argv))

        rc_table = {}
        for mid in ("m6", "m1", "m2", "m8", "m9", "m10", "m4", "m5", "m7",
                    "m11", "m12"):
            rc_table[mid] = _rc_cli("--gaps", mid)
        check("rc: CLEAR is 0", rc_table["m6"] == 0, " | %s" % rc_table["m6"])
        check("rc: BLOCKED is NON-ZERO (it used to be 0)",
              rc_table["m1"] == mi.GAP_RC_BLOCKED, " | %s" % rc_table["m1"])
        check("rc: UNKNOWN is 2 and is not the same code as BLOCKED",
              rc_table["m2"] == mi.GAP_RC_UNKNOWN
              and rc_table["m2"] != rc_table["m1"],
              " | m2=%s m1=%s" % (rc_table["m2"], rc_table["m1"]))
        check("rc: unreadable is 1 and is the ONLY 1",
              _rc_cli("--gaps", "m_no_such_model") == mi.GAP_RC_UNREADABLE
              and rc_table["m8"] == mi.GAP_RC_UNKNOWN,
              " | missing=%s m8=%s" % (_rc_cli("--gaps", "m_no_such_model"),
                                       rc_table["m8"]))
        check("rc: the malformed list no longer shares the code of a missing file",
              rc_table["m8"] != mi.GAP_RC_UNREADABLE, " | m8=%s" % rc_table["m8"])
        check("rc: --require-servable makes it a gate (clear 0, everything else 3)",
              _rc_cli("--gaps", "m6", "--require-servable") == 0
              and _rc_cli("--gaps", "m1", "--require-servable") == 3
              and _rc_cli("--gaps", "m2", "--require-servable") == 3,
              " | clear=%s blocked=%s unknown=%s"
              % (_rc_cli("--gaps", "m6", "--require-servable"),
                 _rc_cli("--gaps", "m1", "--require-servable"),
                 _rc_cli("--gaps", "m2", "--require-servable")))
        # seam 4 的退出码侧: 层型读不出的清单**既是** rc 2 (UNKNOWN) **也是** rc 3 的
        # 门控拒绝 —— 改前这两条都是 rc 0, 而且是带着一份自相矛盾的报告出去的。
        check("rc: an unreadable tier is rc 2, not rc 0",
              rc_table["m11"] == mi.GAP_RC_UNKNOWN
              and rc_table["m12"] == mi.GAP_RC_UNKNOWN,
              " | m11=%s m12=%s" % (rc_table["m11"], rc_table["m12"]))
        check("rc: --require-servable refuses m11/m12 too (it used to answer 0)",
              _rc_cli("--gaps", "m11", "--require-servable") == 3
              and _rc_cli("--gaps", "m12", "--require-servable") == 3,
              " | m11=%s m12=%s" % (_rc_cli("--gaps", "m11", "--require-servable"),
                                    _rc_cli("--gaps", "m12", "--require-servable")))
        # 渲染层: 未测量的报告里, 只许出现"无法判定", 不许出现"可以通过/没有缺口".
        # 对照物用的是 CLEAR 那几条**整句**(而不是"没有缺口"这四个字): UNKNOWN 的
        # 文案里本来就有一句 '"未测量"不等于"没有缺口"', 按词搜会把正确的报告判成红的。
        # 顺带: set_lang() 会把语言写进 gui_lang.txt。自测不该改别人的界面语言偏好,
        # 所以先记下再还原 (实测过: 跑完自测后 gui_lang.txt 从 en 变成 zh)。
        import gui_i18n
        saved_lang = gui_i18n.current()
        for mid in ("m2", "m3", "m4", "m5", "m7", "m8", "m9", "m10", "m11", "m12"):
            for lg in ("zh", "en"):
                rep3 = mi.import_gap_report(mid, lang=lg)   # 顺带设定语言
                txt = mi.render_gap_report_text(rep3)
                clear_only = [mi.t('imp.gaps.no_gaps'), mi.t('imp.gaps.verdict_ok'),
                              mi.t('imp.gaps.header_ok', name="config.h"),
                              mi.t('imp.gaps.line_summary', total=0, hook=0, new_op=0,
                                   post=0)]
                hit = [s for s in clear_only if s and s in txt]
                check("%s/%s renders none of the CLEAR-only lines" % (mid, lg), not hit,
                      " | hit=%s" % [s[:40] for s in hit])
                check("%s/%s prints the UNKNOWN verdict" % (mid, lg),
                      mi.t('imp.gaps.verdict_unmeasured') in txt and "imp.gaps." not in txt,
                      " | %r" % txt[:120])
        gui_i18n.set_lang(saved_lang)

        # ---- seam 5: 头文件那一行是**目录的事实**, 不是结论的文件名形式 ----------
        # 改前是 `"config.h.BLOCKED" if blocking else "config.h"`: 既没 stat 过磁盘,
        # 又断言了一个本树**没有任何写者**产出的名字 (adapt.py 的拒写路径什么都不写,
        # 而 new_op 缺口根本不扣头文件 —— 实测 tools/archkit/out/ornith-1.5-9b-q4-k-m/
        # 有 new_op 缺口**且**有一份 929 B 的 config.h)。这里用两个 fixture 把两侧钉住:
        # m6 的目录里真的放了 config.h, m1 的目录里什么都没放。
        check("seam5: a header that IS next to the manifest is named",
              mi.import_gap_report("m6")["header_file"] == "config.h",
              " | %r" % mi.import_gap_report("m6")["header_file"])
        check("seam5: a blocked manifest with NO header next to it names no file",
              mi.import_gap_report("m1")["header_file"] == "",
              " | %r" % mi.import_gap_report("m1")["header_file"])
        for lg in ("zh", "en"):
            rep1 = mi.import_gap_report("m1", lang=lg)
            txt = mi.render_gap_report_text(rep1)
            check("seam5/%s: the phantom name is gone from the report" % lg,
                  "config.h.BLOCKED" not in txt
                  and mi.t('imp.gaps.header_missing') in txt
                  and "imp.gaps." not in txt, " | %r" % txt[:200])
            rep6 = mi.import_gap_report("m6", lang=lg)
            txt6 = mi.render_gap_report_text(rep6)
            check("seam5/%s: and the header that is there IS named" % lg,
                  mi.t('imp.gaps.header_ok', name="config.h") in txt6,
                  " | %r" % txt6[:200])
        gui_i18n.set_lang(saved_lang)
        # 负对照: 把改前的表达式原样实现一遍, 断言它在这两个 fixture 上给出**不同**
        # 的答案 —— 否则"stat 磁盘"这条规则只是"又一个通过了的用例"。
        def _pre_seam5_header(rep):
            return "config.h.BLOCKED" if rep["blocked"] else "config.h"
        pre5 = {mid: _pre_seam5_header(mi.import_gap_report(mid)) for mid in ("m1", "m6")}
        check("control: the PRE-seam5 expression invented a file for m1",
              pre5["m1"] == "config.h.BLOCKED" and pre5["m6"] == "config.h"
              and pre5["m1"] != mi.import_gap_report("m1")["header_file"],
              " | pre=%s now=%s" % (pre5, mi.import_gap_report("m1")["header_file"]))
        # 负对照之二: 未测量的清单里 header_file 仍然是 "unknown" —— 这一条没有被这次
        # 改动顺手改掉, 因为"没测缺口"确实无法判断头文件能不能用。
        check("control: an unmeasured manifest still says the header is unknown",
              mi.import_gap_report("m2")["header_file"] == "unknown",
              " | %r" % mi.import_gap_report("m2")["header_file"])
    finally:
        mi.ARCHKIT_OUT = saved_out

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
