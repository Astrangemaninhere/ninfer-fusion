#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_compat.py — 多型号 GPU 兼容判定 (新手 GUI 环境卡用)。

回答"我这块显卡能跑 ninfer 吗/能跑哪个档位"。纯标准库, 只读 nvidia-smi。

世代判定依据 (CUDA 13.3 工具链, sm_75 起):
  Tier A  sm_100/120/121  Blackwell/B300  — fp4 (tcgen05/TMA) 原生  -> NVFP4 档全速
  Tier B  sm_89/90        Ada/Hopper      — fp8 TC, 无 fp4        -> 需 FP8 或 Int 档
  Tier C  sm_86/80/75     Ampere/Turing   — int8/int4 TC           -> Groupwise-Int 档
  Tier D  sm_70 及以下    Volta 等        — 不在 CUDA13 工具链内    -> 旧工具链单独构建

注意: 引擎当前构建硬绑定 sm_120a (CMake 门禁), 移植进度见
repo/tools/archkit/_GPU_MATRIX.md; 本模块负责"把话说清楚", 不负责移植本身。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field

# compute capability (major, minor) -> 世代信息
# key 用 major*10+minor; tier 决定"该下载哪个档位的模型/需要哪种构建"
TIERS = {
    120: ("A", "RTX 50 系 (Blackwell)", "fp4 张量核 (tcgen05/TMA) 原生",
          ["nvfp4", "nvfp4-dflash2", "nvfp4-dspark", "groupwise-int"], "NVFP4 全速"),
    121: ("A", "Blackwell 后续", "fp4 张量核原生", ["nvfp4", "groupwise-int"], "NVFP4 全速"),
    100: ("A", "B100/B200 (Blackwell 数据中心)", "fp4 张量核原生",
          ["nvfp4", "groupwise-int"], "NVFP4 全速"),
    110: ("B", "Ampere 数据中心后续", "fp8 张量核", ["groupwise-int"], "Int 档"),
    90:  ("B", "H100/H200 (Hopper)", "fp8 张量核, 无 fp4",
          ["groupwise-int"], "Int 档 (fp8 档规划中)"),
    89:  ("B", "RTX 40 系 (Ada)", "fp8 张量核, 无 fp4",
          ["groupwise-int"], "Int 档 (fp8 档规划中)"),
    86:  ("C", "RTX 30 系 (Ampere)", "int8/int4 张量核",
          ["groupwise-int"], "Groupwise-Int 档"),
    80:  ("C", "A100 (Ampere 数据中心)", "int8/int4 张量核",
          ["groupwise-int"], "Groupwise-Int 档"),
    75:  ("C", "RTX 20 系 (Turing)", "int8/int4 张量核",
          ["groupwise-int"], "Groupwise-Int 档"),
    70:  ("D", "V100 (Volta)", "仅 fp16 张量核; CUDA13 工具链已不支持",
          [], "需 CUDA12 旧工具链单独构建 (最低优先)"),
}
ENGINE_SM120_ONLY = True   # 当前引擎构建硬绑定 sm_120a (见 CMakeLists 门禁)


@dataclass
class GpuCompat:
    present: bool = False
    name: str = ""
    cap: str = ""                       # "12.0" / "8.9" ...
    sm: int = 0                         # major*10+minor
    tier: str = "?"                     # A/B/C/D
    generation: str = "未知"
    features: str = ""
    profiles: list = field(default_factory=list)
    advice: str = ""
    engine_ready: bool = False          # 当前构建能否直接跑 (sm==120)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def query() -> GpuCompat:
    """nvidia-smi 读 compute capability。失败返回 present=False。"""
    c = GpuCompat()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=8)
        if out.returncode != 0 or not out.stdout.strip():
            return c
        parts = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        c.present = True
        c.name = parts[0]
        m = re.match(r"(\d+)\.(\d+)", parts[1])
        if not m:
            return c
        c.cap = parts[1]
        c.sm = int(m.group(1)) * 10 + int(m.group(2))
        tier = TIERS.get(c.sm)
        if tier:
            c.tier, c.generation, c.features, c.profiles, _ = tier
        c.engine_ready = c.sm == 120      # 当前发布的构建只认 sm_120
        # 人话建议
        if not ENGINE_SM120_ONLY or c.sm == 120:
            c.advice = "本机显卡可直接运行 (sm_120 原生构建)。"
        elif c.tier == "A":
            c.advice = "同代 Blackwell: 引擎放开架构门禁后可直接用 NVFP4 档。"
        elif c.tier in ("B", "C"):
            c.advice = ("引擎目前只发布 sm_120 构建; 你的显卡(%s)需要下载 "
                        "%s 档模型 + 对应架构构建 (移植中, 见 _GPU_MATRIX.md)。"
                        % (c.generation, c.profiles[0] if c.profiles else "Int"))
        else:
            c.advice = "该显卡需要旧工具链单独构建, 优先级最低。"
    except Exception:
        pass
    return c


if __name__ == "__main__":
    import json
    print(json.dumps(query().as_dict(), ensure_ascii=False, indent=2))
