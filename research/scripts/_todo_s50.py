#!/usr/bin/env python3
"""Append the S50 landing note (plain concatenation - the text contains % signs)."""
import datetime
import pathlib

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
text = """
### 9. 落地 S50：KV 覆盖改为下界语义（借用上游 03177b9，""" + stamp + """）
- 我方实测：`logical_kv_store.h:1498` 仍是 `if (target < page_count || target > entitlement) throw`
  —— 即"要求覆盖变小"会抛错。上游改成下界语义（已覆盖就早返回、不截断；错误信息带 tokens/pages/entitlement）。
- E7 的可达性分析（其引用行我已复核）：唯一可证明可达的收缩点在 DFlash 终止结算 ——
  verify 已按 `frontier+extent+1` 映射（`program_impl.h:12180`），而 dspark 的 terminal
  `enqueue_dflash_context_append` 只要求 `max(text_kv_valid, end)`（`:11416`，end=base_E+accepted）。
  当"截断的接受 + 跨 64 token 页"同时发生（例 base_E=60 / extent=8 / accepted=2 ⇒ 1<2 页）**今天会抛错**，
  修补后成为无害早返回。⇒ dspark 侧一个真实可触发的失败模式（不解释 10.3% 接受率，但会真炸）。
- 已落地：4 文件 / 26 hunk / +46-38（14 处 `program_impl.h` 调用点改名；E7 用 `diff -u -w -B` 归一化标识符后
  证明参数逐字节不变，我也抽验了引用行）。旧标识符现存 0 处；已 touch 3 个 variant TU 强制重编。
- 未落地：`E7_s50_regression_sketch.diff`（+36 行 store 级页边界回归测试）留到能编 tests 的窗口。
"""
T.open("a", encoding="utf-8").write(text)
print("appended %d chars" % len(text))
