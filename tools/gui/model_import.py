#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_import.py — 模型导入向导的核心识别/判断引擎 (新手友好 GUI 用)。

面向"完全不懂 AI"的使用者。别人给了个模型文件夹/文件,本模块负责回答
三个问题,全部用通俗话术表达:

  1. 这是什么格式?        (.ninfer / GGUF / HuggingFace-safetensors / 其它)
  2. 这台电脑跑得动吗?    (架构是否被 ninfer 支持、显存够不够)
  3. 该怎么处理?          (直接跑 / 自动转换 / 礼貌地说明为什么不支持)

设计原则:
  * 纯标准库 —— GUI/打包/离线全部可用,不依赖 torch。
  * 任何一步失败都降级为"能确定什么说什么",绝不抛异常吓用户。
  * 所有对外文案用中文大白话,数字用 GB 而不是字节。

转换矩阵 (由本模块的 verdict() 给出路线,实际转换由各自 CLI 执行):
  .ninfer                              -> 直接运行
  HF safetensors, 白名单架构, bf16     -> convert.py        (groupwise-int, CPU 可跑)
  HF safetensors, 白名单架构, NVFP4    -> convert_nvfp4.py  (需要 unsloth 预量化源)
  GGUF F16/BF16/F32 (白名单架构)       -> gguf_extract.py   (先抽成 bf16 再走 convert.py)
  GGUF k-quant / 其它架构 / 无法识别   -> 礼貌拒绝 + 建议
"""
from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- 常量

ARTIFACT_MAGIC = b"NINFER\x00\x02"   # ninfer 自研容器格式的文件头
GGUF_MAGIC = b"GGUF"                 # llama.cpp 生态格式的文件头
SUPPORTED_MODEL_IDS = ("qwen3.6-27b", "qwen3.8-27b")  # 引擎 registry 白名单
# GGUF 张量类型 -> 每元素字节(用于体积估算)。来自 llama.cpp gguf 规范。
GGUF_ELEMENT_BYTES = {
    0: 4,    # F32
    1: 2,    # F16
    2: 2,    # BF16
    3: 1,    # Q4_0 (block 量化, 近似 0.5B/参)
    4: 1,    # Q4_1
    5: 2,    # Q5_0
    6: 2,    # Q5_1
    7: 2,    # Q8_0
    8: 1,    # Q8_1
    9: 1,    # Q2_K
    10: 1,   # Q3_K
    11: 1,   # Q4_K
    12: 1,   # Q5_K
    13: 1,   # Q6_K
    14: 1,   # Q8_K
    15: 2,   # IQ2_XXS ...
}
GGUF_QUANT_NAMES = {
    0: "F32 浮点", 1: "F16 半精度", 2: "BF16 半精度", 3: "Q4_0", 4: "Q4_1",
    5: "Q5_0", 6: "Q5_1", 7: "Q8_0", 8: "Q8_1", 9: "Q2_K", 10: "Q3_K",
    11: "Q4_K", 12: "Q5_K", 13: "Q6_K", 14: "Q8_K", 15: "IQ 系列",
}
KQUANT_TYPE_IDS = {9, 10, 11, 12, 13, 14, 15}   # K 系列量化: 目前引擎不支持

# 官方下载链接 (环境自检 / 错误提示时给新手指路)
LINKS = {
    "nvidia_driver": "https://www.nvidia.cn/drivers/",       # 英伟达驱动(中文官网)
    "python": "https://www.python.org/downloads/",            # Python
    "huggingface": "https://huggingface.co/",                 # 模型下载站
}


# ---------------------------------------------------------------- 数据

@dataclass
class GpuInfo:
    """nvidia-smi 探测结果; 探测不到时 present=False。"""
    present: bool = False
    name: str = ""
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    driver: str = ""

    def as_dict(self) -> dict:
        return {"present": self.present, "name": self.name,
                "vram_total_gb": self.vram_total_gb, "vram_free_gb": self.vram_free_gb,
                "driver": self.driver}


@dataclass
class ModelScan:
    """对用户给的一个路径做只读扫描后的结论。"""
    kind: str = "unknown"                 # ninfer | gguf | hf | unknown
    path: str = ""
    friendly: str = ""                    # 一句话格式说明
    model_id: str = ""                    # 白名单 id (ninfer: artifact 身份; hf: 几何匹配)
    weights_id: str = ""                  # ninfer 量化档 (nvfp4-dflash2 等)
    arch_note: str = ""                   # 架构描述 (给人看的)
    params_billions: float = 0.0          # 参数量 (估算)
    weight_bytes: int = 0                 # 权重占用的存储字节 (≈运行内存下限)
    quant_name: str = ""                  # 量化类型描述
    gguf_types: dict = field(default_factory=dict)   # gguf: type_id -> tensor 数
    hidden_size: int = 0                  # hf text_config.hidden_size (几何校验)
    n_layers: int = 0                     # hf text_config.num_hidden_layers
    issues: list = field(default_factory=list)       # 扫描时发现的非致命问题

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Verdict:
    """人类可读的处置建议。action 取值:
    run | convert_groupwise | convert_nvfp4 | need_nvfp4_source |
    vram_short | unsupported_arch | unsupported_quant | unknown"""
    action: str = "unknown"
    level: str = "info"                   # ok | warn | bad
    title: str = ""                       # 大标题, 如 "这个模型可以跑!"
    detail: str = ""                      # 详细话术 (中文大白话)
    tips: list = field(default_factory=list)      # 分条建议


# ---------------------------------------------------------------- 探测

def probe_file_magic(path: Path) -> bytes:
    """读文件头 8 字节, 失败返回 b''。"""
    try:
        with open(path, "rb") as f:
            return f.read(8)
    except OSError:
        return b""


def scan_path(path_str: str) -> ModelScan:
    """入口: 用户给的文件/文件夹路径 -> 完整扫描结论。永远不抛异常。"""
    p = Path(path_str)
    scan = ModelScan(path=str(p))
    try:
        if not p.exists():
            scan.issues.append("路径不存在")
            return scan
        if p.is_file():
            magic = probe_file_magic(p)
            if magic.startswith(ARTIFACT_MAGIC) or p.suffix.lower() == ".ninfer":
                return _scan_ninfer(p)
            if magic.startswith(GGUF_MAGIC):
                return _scan_gguf(p)
            scan.kind = "unknown"
            scan.friendly = "这是一个文件,但不是 ninfer 或 GGUF 模型格式。"
            scan.issues.append("无法识别的文件类型")
            return scan
        # 目录: 先看是不是 safetensors 模型 (config.json + .safetensors)
        cfg = p / "config.json"
        if cfg.is_file() and list(p.glob("*.safetensors")):
            return _scan_hf(p)
        # LoRA 适配器目录 (adapter_config.json, peft_type=LORA)
        acfg = p / "adapter_config.json"
        if acfg.is_file():
            return _scan_lora(p)
        # 目录里可能散放着 .gguf 或 .ninfer
        ggs = list(p.glob("*.gguf"))
        nfs = list(p.glob("*.ninfer"))
        if ggs:
            return _scan_gguf(ggs[0])
        if nfs:
            return _scan_ninfer(nfs[0])
        scan.kind = "unknown"
        scan.friendly = "这个文件夹里没有找到模型文件。请确认里面应该有 "
        scan.friendly += "config.json + .safetensors, 或一个 .gguf / .ninfer 文件。"
        scan.issues.append("目录内无已知模型文件")
        return scan
    except Exception as e:  # 任何意外都降级, 不 crash
        scan.issues.append("扫描出错: %s" % e)
        return scan


def _read_hf_config(p: Path) -> dict:
    try:
        with open(p / "config.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _scan_ninfer(p: Path) -> ModelScan:
    """ninfer 自研容器: 尝试读身份 (model_id/weights_id), 读不到也不致命。"""
    s = ModelScan(kind="ninfer", path=str(p), friendly="这是 ninfer 自家格式,可以直接运行。")
    try:
        # 只流式读文件头 1MB (容器目录 JSON 在头部); 绝不能用 read_bytes()——
        # 十几 GB 的 artifact 会直接把内存吃爆。
        with open(p, "rb") as f:
            head = f.read(1 << 20)
        m = re.search(rb'"model_id"\s*:\s*"([^"]+)"', head)
        w = re.search(rb'"weights_id"\s*:\s*"([^"]+)"', head)
        if m:
            s.model_id = m.group(1).decode("utf-8", "replace")
        if w:
            s.weights_id = w.group(1).decode("utf-8", "replace")
        s.weight_bytes = p.stat().st_size
        s.params_billions = round(s.weight_bytes / (1 << 30) * 1.0, 2)  # 粗略(含 KV 表等)
        s.quant_name = "NVFP4/自研量化"
    except OSError:
        s.issues.append("文件读取失败")
    return s


def _scan_lora(p: Path) -> ModelScan:
    """LoRA/PEFT 适配器目录: 本身不能跑, 需基座 + 离线融合。"""
    s = ModelScan(kind="lora", path=str(p),
                  friendly="这是 LoRA 微调包(只含增量权重,不是完整模型)。")
    try:
        with open(p / "adapter_config.json", encoding="utf-8") as f:
            ac = json.load(f)
        s.quant_name = "LoRA r=%s" % ac.get("r", "?")
        base = ac.get("base_model_name_or_path") or ac.get("base_model_name_or_path", "")
        s.arch_note = ("PEFT %s; 基座: %s" % (ac.get("peft_type", "?"), base)) \
            if base else "PEFT %s; 基座信息缺失" % ac.get("peft_type", "?")
        total = sum(f.stat().st_size for f in p.glob("*.safetensors"))
        s.weight_bytes = total
        s.issues.append("LoRA 需对应基座模型 + 离线融合(见 tools/convert/lora/_LORA.md)")
    except OSError as e:
        s.issues.append("adapter_config.json 读取失败: %s" % e)
    return s


def _scan_hf(p: Path) -> ModelScan:
    """HuggingFace 风格目录 (vLLM 也吃这种): config.json + safetensors。"""
    s = ModelScan(kind="hf", path=str(p),
                  friendly="这是 HuggingFace/safetensors 格式(vLLM 也用它)。")
    cfg = _read_hf_config(p)
    archs = cfg.get("architectures") or []
    mt = cfg.get("model_type", "")
    # 多模态模型会把文本部分几何放在 text_config 里
    tc = cfg.get("text_config") or {}
    s.hidden_size = int(tc.get("hidden_size") or cfg.get("hidden_size") or 0)
    s.n_layers = int(tc.get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)
    s.arch_note = ("架构: %s (%s)" % (", ".join(archs) or "未知", mt)) if mt else "架构: 未知"
    s.model_id = _family_from_arch(archs, mt, s.hidden_size, s.n_layers)
    if s.model_id == "qwen3-candidate":
        s.arch_note += " —— Qwen3 家族, 但不是软件支持的规格"
    elif not s.model_id:
        s.arch_note += " —— 这个架构不在本软件支持列表里。"
    # 参数量/权重体积: 直接按文件大小 (bf16 约 2 字节/参数)
    total = 0
    try:
        for f in p.glob("*.safetensors"):
            total += f.stat().st_size
    except OSError:
        pass
    s.weight_bytes = total
    s.params_billions = round(total / 2 / 1e9, 2) if total else 0.0
    s.quant_name = _hf_quant_hint(cfg)
    return s


# 引擎 TextConfig 几何 (qwen3_6_27b target): 只有这个规格的 Qwen3 才能跑
ENGINE_HIDDEN = 5120
ENGINE_LAYERS = 64


def _family_from_arch(archs: list, model_type: str, hidden: int, layers: int) -> str:
    """HF 架构 -> 支持结论。

    返回:
      'qwen3.8-27b'  完全匹配支持规格 (Qwen3 家族 + 5120 隐藏 + 64 层)
      'qwen3-candidate'  是 Qwen3 家族但几何不是支持规格
      ''             完全不在家族
    """
    joined = " ".join([*archs, model_type]).lower().replace("_", "")
    if "qwen3" not in joined and "qwen2.5" not in joined:
        return ""
    if (hidden == ENGINE_HIDDEN and layers == ENGINE_LAYERS) or hidden == 0:
        # 几何一致, 或 config 里读不到几何(名字很像) -> 按候选给, 转换器会做最终校验
        return "qwen3.8-27b" if "qwen3" in joined else "qwen3-candidate"
    return "qwen3-candidate"


def _hf_quant_hint(cfg: dict) -> str:
    """看 config 里的蛛丝马迹猜量化档, 只用于文案。"""
    qc = str(cfg.get("quantization_config") or {})
    if "nvfp4" in qc.lower() or "fp4" in qc.lower():
        return "NVFP4 预量化(推荐, 转换后体积小)"
    if "bitsandbytes" in qc.lower():
        return "bitsandbytes 量化"
    return "BF16/FP16 全精度(转换后体积大)"


def _scan_gguf(p: Path) -> ModelScan:
    """llama.cpp GGUF: 解析文件头 + 张量表, 判断量化档与架构家族。"""
    s = ModelScan(kind="gguf", path=str(p), friendly="这是 GGUF 格式(llama.cpp 生态)。")
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                s.issues.append("GGUF 文件头不完整")
                return s
            version, n_tensors, n_kv = struct.unpack("<IQI", f.read(16))
            s.friendly = "GGUF 格式 (版本 %d, %d 个张量)" % (version, n_tensors)
            # 跳过 metadata KV 区: 每个 entry = key(str) + type(u32) + value
            for _ in range(n_kv):
                _skip_str(f)
                ktype = struct.unpack("<I", f.read(4))[0]
                _skip_gguf_value(f, ktype)
            # 张量表: name(str) + shape(u32 个数 + 每维 i64) + type(u32) + offset(i64)
            total = 0
            arch = ""
            for _ in range(n_tensors):
                _skip_str(f)                       # tensor name
                ndims = struct.unpack("<I", f.read(4))[0]
                f.read(8 * ndims)                  # shape
                ttype = struct.unpack("<I", f.read(4))[0]
                f.read(8)                          # offset
                eb = GGUF_ELEMENT_BYTES.get(ttype, 1)
                s.gguf_types[ttype] = s.gguf_types.get(ttype, 0) + 1
                total += eb
            # 大致的参数量: GGUF 文件里主权重占绝对大头, 用文件大小兜底更稳
            s.weight_bytes = p.stat().st_size
            s.params_billions = round(s.weight_bytes / 1e9, 2)  # 粗估(含量化压缩)
            names = [GGUF_QUANT_NAMES.get(t, "未知(%d)" % t)
                     for t in sorted(s.gguf_types)]
            s.quant_name = "/".join(names) if names else "未知"
            if any(t in KQUANT_TYPE_IDS for t in s.gguf_types):
                s.issues.append("包含 K 系列量化(K-quant),引擎尚不支持")
            # 架构: 只能从文件名猜家族 (GGUF 内部有 general.architecture KV,
            # 但可能被跳过; 用文件名关键词兜底)
            s.model_id = _family_from_gguf_name(p.name)
            if not s.model_id:
                s.arch_note = "无法从文件名确认架构家族"
    except (OSError, struct.error) as e:
        s.issues.append("GGUF 解析失败: %s" % e)
    return s


def _family_from_gguf_name(name: str) -> str:
    low = name.lower().replace("_", "")
    if "qwen3" in low or "qwen2.5" in low:
        # GGUF 里读不到几何, 只能给候选; 最终由转换器预检把关
        return "qwen3.8-27b" if "qwen3" in low else "qwen3-candidate"
    return ""


def _skip_str(f):
    """GGUF 字符串: u64 长度 + 字节。"""
    n = struct.unpack("<Q", f.read(8))[0]
    f.read(n)


def _skip_gguf_value(f, ktype: int, depth: int = 0):
    """跳过 GGUF metadata value。类型表来自 gguf 规范 0..13。
    深度/数量都设上限,防恶意或损坏文件拖死扫描。"""
    if depth > 4:
        raise struct.error("metadata nesting too deep")
    if ktype == 0:          # uint8
        f.read(1)
    elif ktype == 1:        # int8
        f.read(1)
    elif ktype == 2:        # uint16
        f.read(2)
    elif ktype == 3:        # int16
        f.read(2)
    elif ktype == 4:        # uint32
        f.read(4)
    elif ktype == 5:        # int32
        f.read(4)
    elif ktype == 6:        # float32
        f.read(4)
    elif ktype == 7:        # bool
        f.read(1)
    elif ktype == 8:        # string
        _skip_str(f)
    elif ktype == 9:        # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        if count > (1 << 20):
            raise struct.error("metadata array too large")
        for _ in range(count):
            _skip_gguf_value(f, elem_type, depth + 1)
    elif ktype == 10:       # uint64
        f.read(8)
    elif ktype == 11:       # int64
        f.read(8)
    elif ktype == 12:       # float64
        f.read(8)
    else:
        raise struct.error("unknown metadata type %d" % ktype)


# ---------------------------------------------------------------- 判断

def gpu_report() -> GpuInfo:
    """nvidia-smi 探测显卡。没有 NVIDIA 卡/驱动时 present=False。"""
    info = GpuInfo()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8)
        if out.returncode != 0 or not out.stdout.strip():
            return info
        parts = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        info.present = True
        info.name = parts[0]
        info.vram_total_gb = round(float(parts[1]) / 1024, 1)
        info.vram_free_gb = round(float(parts[2]) / 1024, 1)
        info.driver = parts[3] if len(parts) > 3 else ""
    except Exception:
        pass
    return info


def vram_need_bytes(scan: ModelScan, ctx_tokens: int = 8192) -> int:
    """运行所需显存的粗略下界。

    组成: 权重驻留(转换后约 0.55 字节/参数量级) + KV 缓存 + 计算暂存。
    精确值由引擎 arena 决定; 这里是给新手看的"装不装得下"判据。
    """
    if scan.kind == "ninfer":
        # ninfer 权重本身常驻; KV 与 scratch 粗估按上下文每 token ~0.4MB
        return int(scan.weight_bytes + ctx_tokens * 400_000)
    # bf16 源权重转换后按 ~0.55x 估算 (NVFP4 每参数约 4.2 bit + scale)
    est_weights = int(scan.weight_bytes * 0.55)
    return int(est_weights + ctx_tokens * 400_000)


def verdict(scan: ModelScan, gpu: Optional[GpuInfo] = None,
            ctx_tokens: int = 8192) -> Verdict:
    """把扫描结果翻译成新手能读懂的处置建议。"""
    gpu = gpu or gpu_report()
    v = Verdict()
    if scan.kind == "ninfer":
        v.action, v.level = "run", "ok"
        v.title = "这个模型可以直接运行!"
        v.detail = ("这是 ninfer 自家的模型文件(%s),不用转换。"
                    % (scan.quant_name or "已量化"))
        if scan.model_id and scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = "这个模型文件本软件跑不了"
            v.detail = "文件格式是对的,但它的模型家族(%s)不在支持列表里。" % scan.model_id
        if gpu.present:
            need = vram_need_bytes(scan, ctx_tokens)
            if need > gpu.vram_total_gb * (1 << 30):
                v.action, v.level = "vram_short", "bad"
                v.title = "显存可能不够,建议先试试小上下文"
                v.detail = ("模型本体约 %.1f GB,你的显卡 %s 只有 %.1f GB 显存,"
                            "开 8192 上下文可能放不下。可以先减小上下文(比如 2048)再试。"
                            % (scan.weight_bytes / 1e9, gpu.name, gpu.vram_total_gb))
                v.tips = ["把上下文长度调小(2048~4096)", "换更小的模型",
                          "购买更大显存的显卡(本软件最低建议 16GB+)"]
        else:
            v.level = "warn"
            v.tips.append("没有检测到 NVIDIA 显卡:本软件需要 NVIDIA 显卡(CUDA)才能跑。")
            v.tips.append("驱动安装: %s" % LINKS["nvidia_driver"])
        return v
    if scan.kind == "hf":
        if scan.model_id == "qwen3-candidate":
            v.action, v.level = "unsupported_arch", "bad"
            v.title = "这是 Qwen3 家族,但不是软件支持的规格"
            v.detail = ("本软件目前只支持 5120 隐藏宽度、64 层的 27B 版 Qwen3。"
                        "你这份模型(%d 层 / %d 隐藏)对不上,转换出来也跑不了,"
                        "所以就不浪费时间转换了。" % (scan.n_layers, scan.hidden_size))
            v.tips.append("去 %s 下载 Qwen3.8-27B(27B, 64 层)或 Qwen3.6-27B。" % LINKS["huggingface"])
            return v
        if scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = "这个模型本软件跑不了"
            v.detail = ("你下载的模型架构不在本软件支持列表里。目前只支持 Qwen3 家族的 "
                        "27B/64 层版本。支持列表外的模型需要专门的适配(每个架构的推理引擎"
                        "都要重新实现),不是简单转换能解决的。")
            v.tips.append("去 %s 搜索支持列表里的模型下载。" % LINKS["huggingface"])
            return v
        v.action = "convert_groupwise"
        v.level = "ok"
        v.title = "可以自动转换成 ninfer 格式再运行"
        v.detail = "这个模型(%s)属于支持家族,可以转换。转换需要一些时间(纯 CPU 也能跑)。" % scan.model_id
        need = vram_need_bytes(scan, ctx_tokens)
        if gpu.present and need > gpu.vram_total_gb * (1 << 30) * 0.92:
            v.action, v.level = "vram_short", "bad"
            v.title = "转换完也装不下:显存不够"
            v.detail = ("这个模型 %s 约 %.1f 亿参数,转换后仍需 %.1f GB 左右显存,"
                        "你的显卡只有 %.1f GB。建议换小一号的模型,或换大显存显卡。"
                        % (scan.quant_name, scan.params_billions * 10,
                           need / 1e9, gpu.vram_total_gb))
            v.tips = ["模型越大越吃显存: 27B 级别建议 24GB+ 显卡",
                      "Qwen 系列有小号版本可下载", "安装/更新驱动: %s" % LINKS["nvidia_driver"]]
            return v
        if "NVFP4" not in scan.quant_name:
            v.tips.append("提示:全精度版转换后体积大;有 NVFP4 预量化版的话下载那个更快。")
        return v
    if scan.kind == "gguf":
        if any(t in KQUANT_TYPE_IDS for t in scan.gguf_types):
            v.action, v.level = "unsupported_quant", "bad"
            v.title = "这个 GGUF 是 K 系列量化,暂时不支持"
            v.detail = ("K 系列量化是 llama.cpp 的压缩格式,本软件不能直接读。"
                        "请到模型页面下载 F16 或 BF16(未量化)版本,文件名通常带 "
                        "'f16' 或 'bf16' 字样,后缀仍是 .gguf。")
            return v
        if scan.model_id == "qwen3-candidate" or scan.model_id not in SUPPORTED_MODEL_IDS:
            v.action, v.level = "unsupported_arch", "bad"
            v.title = "这个模型本软件跑不了"
            v.detail = "这个 GGUF 对应的模型家族不在支持列表里(只支持 Qwen3 的 27B/64 层版本)。"
            return v
        v.action = "convert_groupwise"
        v.level = "ok"
        v.title = "可以转换成 ninfer 格式再运行"
        v.detail = "GGUF 会先解包成半精度,再转成 ninfer 格式,需要一些时间。"
        v.tips.append("只支持 F16/BF16/F32 的 GGUF; K-quant 版本请换源。")
        return v
    if scan.kind == "lora":
        v.action, v.level = "need_base_fuse", "warn"
        v.title = "这是 LoRA 微调包,还需要一步融合"
        v.detail = ("LoRA 只含增量权重(rank %s),不能单独运行。需要: "
                    "① 对应的基座模型(见 adapter_config 里的 base_model), "
                    "② 用融合工具把它离线合进基座,生成一个新的 .ninfer。"
                    "融合是逐层 解量化→加增量→重量化,需要一些时间。"
                    % (scan.quant_name or "?"))
        v.tips.append("工具与流程: tools/convert/lora/lora_merge.py + _LORA.md")
        return v
    v.action = "unknown"
    v.level = "bad"
    v.title = "没能认出这个模型"
    v.detail = "请确认给的是一个 .ninfer / .gguf 文件,或一个含 config.json 的模型文件夹。"
    return v


def env_report() -> dict:
    """环境自检: 显卡/驱动/Python/引擎产物。给新手看 + 打包后首启检查用。"""
    gpu = gpu_report()
    items = []
    if gpu.present:
        items.append({"name": "NVIDIA 显卡", "ok": True,
                      "detail": "%s, 显存 %.1f GB, 驱动 %s" % (gpu.name, gpu.vram_total_gb, gpu.driver)})
    else:
        items.append({"name": "NVIDIA 显卡", "ok": False,
                      "detail": "没有检测到。本软件需要 NVIDIA 显卡 + 驱动。",
                      "fix": "安装驱动: " + LINKS["nvidia_driver"]})
    items.append({"name": "Python", "ok": True,
                  "detail": sys.version.split()[0]})
    # 引擎产物: 优先 Windows 侧 exe (打包后), 否则 WSL ninfer-serve (开发期)
    engine = None
    for cand in (Path(sys.executable).parent / "ninfer-serve.exe",
                 Path("ninfer-serve.exe")):
        if cand.exists():
            engine = str(cand)
            break
    if engine is None:
        try:
            r = subprocess.run(["wsl.exe", "-d", "Ubuntu", "bash", "-lc",
                                "ls /home/user/ninfer-fusion/build/apps/ninfer-serve 2>/dev/null"],
                               capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                engine = "wsl:" + r.stdout.strip()
        except Exception:
            pass
    items.append({"name": "推理引擎 ninfer-serve", "ok": engine is not None,
                  "detail": engine or "未找到(Windows 版打包后自动内置)",
                  "fix": None if engine else "开发环境需先在 WSL 构建"})
    return {"items": items}


if __name__ == "__main__":
    # 命令行自测: python model_import.py <路径>
    for path in sys.argv[1:]:
        sc = scan_path(path)
        v = verdict(sc)
        print("== %s" % path)
        print("  格式: %s | %s" % (sc.kind, sc.friendly))
        if sc.model_id:
            print("  家族: %s" % sc.model_id)
        if sc.weight_bytes:
            print("  体积: %.2f GB | %s" % (sc.weight_bytes / 1e9, sc.quant_name or "-"))
        if sc.issues:
            print("  注意: %s" % "; ".join(sc.issues))
        print("  结论[%s]: %s" % (v.level, v.title))
        print("  %s" % v.detail)
        for t in v.tips:
            print("  - %s" % t)
