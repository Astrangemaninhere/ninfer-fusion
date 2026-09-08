#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hwdec.py — 显卡硬解能力判定 (从 GPU 本身取信息, 不依赖显卡名字白名单).

数据源 (全部来自 GPU/driver 本体):
  1. pynvml: name / brand / 架构代次 (NVML_ARCH_*)  —— 未知卡也能报真实身份
  2. nvidia-smi --query-gpu=compute_cap  —— 算力代次
判定规则 (NVDEC 硬解 = 架构代次 + 驱动在线; 架构代次来自 GPU 硬件自身):
  Turing (10.x)+  => NVDEC 硬解可用 (h264/hevc/vp9 起步; Turing 加 AV1 解码? AV1
     解码自 Ampere (8.x) 起)
  Kepler..Pascal  => NVDEC 可用但老编解码有限 (h264/hevc)
  非 NVIDIA / 无卡 => 仅软解
用户仍可强制选择软解 (GUI 选项永远开放).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field


@dataclass
class HwDecInfo:
    present: bool = False
    vendor: str = ""            # nvidia / unknown
    name: str = ""              # GPU 报的真实名字 (未知识也能显示)
    arch: str = ""              # NVML 架构名 (Pascal/Volta/Turing/...)
    compute_cap: str = ""       # "8.9"
    gen: int = 0                # 架构代次整数 (maj*10+min), 用于 NVDEC 矩阵
    hw_decoders: list = field(default_factory=list)   # ["h264","hevc",...]
    hw_ok: bool = False
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "present": self.present, "vendor": self.vendor, "name": self.name,
            "arch": self.arch, "compute_cap": self.compute_cap, "gen": self.gen,
            "hw_decoders": self.hw_decoders, "hw_ok": self.hw_ok, "reason": self.reason,
        }


# NVML 架构枚举 (pynvml.nvmlDeviceGetArchitecture 返回值)
NVML_ARCH = {
    0: "Unknown", 1: "Kepler", 2: "Maxwell", 3: "Maxwell",
    4: "Pascal", 5: "Volta", 6: "Turing", 7: "Ampere", 8: "Ada", 9: "Hopper",
    10: "Blackwell", 11: "Blackwell+",
}
# 架构 -> NVDEC 解码能力 (按硬件解码引擎代次, 不看市场名)
ARCH_DECODE = {
    "Kepler":   ["h264"],
    "Maxwell":  ["h264", "hevc"],
    "Pascal":   ["h264", "hevc", "vp9"],
    "Volta":    ["h264", "hevc", "vp9"],
    "Turing":   ["h264", "hevc", "vp9"],
    "Ampere":   ["h264", "hevc", "vp9", "av1"],
    "Ada":      ["h264", "hevc", "vp9", "av1"],
    "Hopper":   ["h264", "hevc", "vp9", "av1"],
    "Blackwell": ["h264", "hevc", "vp9", "av1"],
    "Blackwell+": ["h264", "hevc", "vp9", "av1"],
}


def probe() -> HwDecInfo:
    info = HwDecInfo()
    # 1) NVML: GPU 本体身份 + 架构 (未知卡也返回真实信息)
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        info.name = name.decode() if isinstance(name, bytes) else str(name)
        brand = pynvml.nvmlDeviceGetBrand(h)
        info.vendor = "nvidia" if 0 <= int(brand) <= 16 else "unknown"
        try:
            arch_id = int(pynvml.nvmlDeviceGetArchitecture(h))
            info.arch = NVML_ARCH.get(arch_id, f"NVML-{arch_id}")
        except Exception:
            info.arch = ""
        info.present = True
    except Exception as e:
        info.reason = "pynvml 不可用: " + str(e)[:80]
        return info

    if info.vendor != "nvidia":
        info.reason = "非 NVIDIA 卡: 仅软解"
        return info

    # 2) compute_cap (driver 报告)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        info.compute_cap = out.stdout.strip().splitlines()[0].strip()
        m = re.match(r"(\d+)\.(\d+)", info.compute_cap)
        if m:
            info.gen = int(m.group(1)) * 10 + int(m.group(2))
    except Exception:
        pass

    # 3) NVDEC 矩阵: 优先架构名 (GPU 硬件自身), 缺失时按 compute_cap 代次
    dec = ARCH_DECODE.get(info.arch)
    if dec is None and info.gen:
        if info.gen >= 80:
            dec = ARCH_DECODE["Ampere"]
        elif info.gen >= 60:
            dec = ARCH_DECODE["Pascal"]
        else:
            dec = ARCH_DECODE["Kepler"]
    if dec:
        info.hw_decoders = dec
        info.hw_ok = True
        info.reason = f"NVDEC 可用 ({info.arch or 'gen %d' % info.gen})"
    else:
        info.reason = "架构过老或未知: 仅软解"
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(probe().as_dict(), ensure_ascii=False, indent=1))
