#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_compat.py — 显卡身份 / 路线说明卡 (新手 GUI 环境卡用)。

回答"我这张卡是什么"; 纯标准库, 只读 nvidia-smi。

它 **不回答** "我这张卡当前能不能跑" —— 因为那个答案不在 arch 号里。
本机实测 (RTX 5090 D): 同一个 arch 号 120, 120a 构建报六种 KV 格式全部
SUPPORTED, 120 构建拒 fp8/nvfp4。原因是 nvcc 只为 'a' (arch-specific) 目标
定义 __CUDA_ARCH_FEAT_SM120_ALL, fp4/TMA 内核只在那个宏下存在; 而
`device.sm()` (= major*10+minor) 看不到这个区别。所以:

  * `engine_ready` 恒为 None = **未知**。本模块不允许由 sm 号推断它, 也不允许
    把它渲染成"能跑"。它只能由引擎自己的能力探针填写 (真实构建报出哪些 KV
    格式被接受)。GUI 侧看到 None 就该说"未知, 见引擎探针", 而不是打勾。
  * `TIERS` 是**未验证的移植路线规划** (哪个内核族面向哪一代卡), 不是能力
    判定; 任何一行都不构成"这个模型能跑"的说法。之前那一版把
    `engine_ready = (sm == 120)` 写死在这里, 就是这个缺陷本身。

世代命名 (CUDA 13 工具链, compute capability 75 起):
  sm_100/120/121  Blackwell/B300  — fp4 (tcgen05/TMA) 路线
  sm_89/90        Ada/Hopper      — fp8 TC, 无 fp4
  sm_86/80/75     Ampere/Turing   — int8/int4 TC, groupwise-int 路线
  sm_70 及以下    Volta 等        — 不在 CUDA 13 工具链内

移植/构建现状见 repo/tools/archkit/_GPU_MATRIX.md。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# 路线规划表 (逐行未验证, 见模块头)。key = major*10+minor。
# 元组 = (tier, 世代名, 该代目标内核族, 该代规划的权重档, 路线一句话)
TIERS = {
    120: ("A", "RTX 50 系 (Blackwell)", "fp4 张量核 (tcgen05/TMA) 路线",
          ["nvfp4", "nvfp4-dflash2", "nvfp4-dspark", "groupwise-int"],
          "NVFP4 路线 (未验证)"),
    121: ("A", "Blackwell 后续", "fp4 张量核路线", ["nvfp4", "groupwise-int"],
          "NVFP4 路线 (未验证)"),
    100: ("A", "B100/B200 (Blackwell 数据中心)", "fp4 张量核路线",
          ["nvfp4", "groupwise-int"], "NVFP4 路线 (未验证)"),
    110: ("B", "Ampere 数据中心后续", "fp8 张量核路线", ["groupwise-int"],
          "Int 路线 (未验证)"),
    90:  ("B", "H100/H200 (Hopper)", "fp8 张量核, 无 fp4", ["groupwise-int"],
          "Int 路线 (fp8 档未落地)"),
    89:  ("B", "RTX 40 系 (Ada)", "fp8 张量核, 无 fp4", ["groupwise-int"],
          "Int 路线 (fp8 档未落地)"),
    86:  ("C", "RTX 30 系 (Ampere)", "int8/int4 张量核", ["groupwise-int"],
          "Groupwise-Int 路线 (未验证)"),
    80:  ("C", "A100 (Ampere 数据中心)", "int8/int4 张量核", ["groupwise-int"],
          "Groupwise-Int 路线 (未验证)"),
    75:  ("C", "RTX 20 系 (Turing)", "int8/int4 张量核", ["groupwise-int"],
          "Groupwise-Int 路线 (未验证)"),
    70:  ("D", "V100 (Volta)", "仅 fp16 张量核", [],
          "需 CUDA 12 旧工具链单独构建 (未验证)"),
}

# "这个结论是谁给的" —— 唯一被允许填 engine_ready 的来源。
CAPABILITY_SOURCE_ENGINE_PROBE = "engine-probe"
CAPABILITY_SOURCE_UNKNOWN = "unknown"


@dataclass
class GpuCompat:
    present: bool = False
    name: str = ""
    cap: str = ""                       # "12.0" / "8.9" ...
    sm: int = 0                         # major*10+minor
    tier: str = "?"                     # A/B/C/D (路线规划, 未验证)
    generation: str = "未知"
    features: str = ""                  # 该代**规划**的内核族, 未验证
    profiles: list = field(default_factory=list)   # 该代**规划**的权重档, 未验证
    route_note: str = ""                # 路线一句话, 未验证
    # None = 未知。唯一合法来源是引擎自己的能力探针; sm 号决定不了它。
    engine_ready: Optional[bool] = None
    engine_ready_reason: str = ""
    capability_source: str = CAPABILITY_SOURCE_UNKNOWN

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def query() -> GpuCompat:
    """nvidia-smi 读名字与 compute capability。失败返回 present=False。

    只填"这块卡是什么"。engine_ready 恒为 None: 本进程没有引擎探针结果,
    而 sm 号不是证据 (见模块头)。
    """
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
            c.tier, c.generation, c.features, c.profiles, c.route_note = tier
        # 这里没有"从号推能力"这一步可写: 同一个 sm 号 (120) 下 120a 与 120
        # 构建的结论不同, 所以答案只能是"未知, 待探针"。
        c.engine_ready = None
        c.engine_ready_reason = (
            "未知: 需引擎自身的能力探针 (真实构建报出的 KV 格式集)。sm%s 决定不了它 —— "
            "同为 sm120 的 120a 构建报六种格式全支持, 120 构建拒 fp8/nvfp4。" % c.sm)
        c.capability_source = CAPABILITY_SOURCE_UNKNOWN
        # 人话建议: 只说事实与下一步, 不说"能跑"。
        route = ("本代规划路线: %s; 需配套构建 (%s), 见 _GPU_MATRIX.md。"
                 % (c.features, c.route_note) if c.features else c.route_note)
        c.advice = ("引擎能否直接跑本卡以引擎的能力探针为准 (本模块不推断)。" + route)
    except Exception:
        pass
    return c


# ---------------------------------------------------------------------------
# THE THREE STATES OF A CAPABILITY CLAIM, and why the third one is the usual answer.
#
#   CAPABILITY_MEASURED    the engine's OWN known-answer capability probe reported this build's
#                          accepted KV format set on this card. The only thing that is support.
#   CAPABILITY_REFUSED     measured, and this build refuses the card.
#   CAPABILITY_UNMEASURED  nobody measured it. A route plan (TIERS), a build number, an arch
#                          number, a simulator run -- none of these is a measurement, and none
#                          of them may be rendered as a pass.
#
# This exists because the environment card in ninfer-gui.py used to stamp `ok: True` on the
# GPU-generation row as a CONSTANT, and gui_page.html renders `ok` as a green tick in a list
# where every other green tick means "this self-check requirement is satisfied". The module
# below had already refused to infer the answer -- `engine_ready` is None and
# `capability_source` is "unknown" -- but NOTHING read either field, so the honest refusal was
# decoration and the constant was the verdict. A field whose value no instrument produced must
# not be rendered as a pass.
# ---------------------------------------------------------------------------
CAPABILITY_MEASURED = "measured"
CAPABILITY_REFUSED = "refused"
CAPABILITY_UNMEASURED = "unmeasured"


def capability_state(compat: dict) -> str:
    """MEASURED / REFUSED / UNMEASURED, decided by TWO fields together and fail-closed.

    `engine_ready is True` alone is not enough: a future writer could set the boolean without
    an instrument behind it, which is the exact defect this replaces. The source has to say
    "engine-probe" as well, so both the value and its provenance must be present.
    """
    if compat.get("capability_source") != CAPABILITY_SOURCE_ENGINE_PROBE:
        return CAPABILITY_UNMEASURED
    if compat.get("engine_ready") is True:
        return CAPABILITY_MEASURED
    if compat.get("engine_ready") is False:
        return CAPABILITY_REFUSED
    return CAPABILITY_UNMEASURED


def env_item(compat: dict) -> dict:
    """The environment-card row for a GPU. `ok` is True/False/None; None means NOT MEASURED.

    gui_page.html renders the three states as ✔ / ✖ / ? -- never a tick for an unmeasured
    claim, and never a cross either, because "your card cannot run this" is also a verdict and
    is just as unmeasured as its opposite.
    """
    state = capability_state(compat)
    ok = {"measured": True, "refused": False}.get(state)   # None for unmeasured
    detail = "%s — %s (%s)" % (compat.get("generation", "?"), compat.get("features", ""),
                               compat.get("advice", ""))
    if state == CAPABILITY_UNMEASURED:
        detail += (" [未测量: 本行是路线规划, 不是能力判定; 引擎能否跑本卡只有引擎自己的"
                   "能力探针能回答]")
    return {"name": "显卡世代", "ok": ok, "detail": detail,
            "fix": None, "capability_state": state}


if __name__ == "__main__":
    import json
    print(json.dumps(query().as_dict(), ensure_ascii=False, indent=2))
