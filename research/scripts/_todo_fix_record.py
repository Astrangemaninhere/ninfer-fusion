#!/usr/bin/env python3
"""Correct the record: my S45c 'silent skip at 256' claim was wrong (E3 was right),
and land S45d instead."""
import datetime
import pathlib

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
T.open("a", encoding="utf-8").write(f"""
### 7. 更正（{stamp}）：我关于 S45c 的"256 静默回归"判断是**错的**
上节说 E3 的 S45c 守卫"合并条件导致 ISO3/FP8 在 256 下静默跳过"，这个结论**不成立**，E3 的申辩是对的。
实证方法：把 `E3_s45c_prefill_guard.diff.SUPERSEDED` 打到 pristine 的临时副本上，做**括号深度追踪**
（`_s45c_depth.py`）——合并分支 `}} else if (NVFP4 || ISO3 || FP8) {{` 在 depth 4，而
`}} else if (cache.dtype == DType::ISO3)` 在 **depth 6**、FP8 同在 depth 6，说明这两条臂被**串进了守卫内部**
（`if constexpr (256) {{ if (NVFP4) … else if (ISO3) … else if (FP8) {{ … }} else {{ … }} }}`），
**256 下是可到达的**。我此前的读法只看 diff hunk，而这两条 `}} else if` 分隔行在补丁里是上下文行、不在 hunk 内，
于是被误读成"外层死代码"。
S45c 真正的缺口只有一个：128 下 packed 臂**没有响亮抛错**（会静默什么都不做）——那是防御性缺口，不是 256 回归。
因此当时的回退并非必需（无害：树回到 pristine 约 10 分钟，期间没有任何测量基于它）。

**已落地**：`E3_s45d_prefill_guard_v2.diff`（逐臂守卫、保留原条件、本体逐字节不变）。
我独立复核过 E3 的证据：`removed/changed = 0`、`added = 31`、453→484 行、三条 dtype 条件出现次数不变、
每臂各自 `if constexpr + else-throw`（114/168、173/196、201/224）、括号深度收于 0。
attr 组守卫保持"无 else"（对 bf16/i8 也执行，加了 else 会把 Muse 唯一可用的 prefill 档也毙掉）。
`E3_s45d_s36_restore.md` 记录 S36 派生恢复是**单行值替换、不含控制流、与 S45d 行不重叠**，
128 下只是 fill grid 放大 2 倍（性能），继续推迟。
""" )
print("correction written")
