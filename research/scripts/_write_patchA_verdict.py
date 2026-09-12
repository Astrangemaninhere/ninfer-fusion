#!/usr/bin/env python3
"""落盘：补丁 A 的首批实测数据 + 判读 + 一个新发现的陷阱（浮点数口径）到 _collab 与 _TODO.md。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
M = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/M_patchA_effect.md")
M.open("a", encoding="utf-8").write("""

---

## 判读（M，""" + stamp + """）—— 补丁 A 首批实测

### 1. 已修掉的致命症状
修复前：dflash2 在同一 prompt 上 `gen=2` 立即停（verify 第 0 列 argmax 是错的，a=0 时只发布那个 token）。
修复后：`dflash2_zh gen=83` / `dflash2_num gen=76`（plain 对照 96/88）⇒ **立即停止消失，输出长度回到正常量级**。

### 2. 接受率（三档，同一 prompt、同一温度、同一新二进制）
| 后端 | prompt | gen | 接受率(accepted/drafted) | rounds | fallback steps | decode |
|---|---|---|---|---|---|---|
| dflash2 | zh (22 tok) | 83 | **6/133 = 4.51%** | 19 | **57** | 17.6 tok/s |
| dflash2 | num (130 tok) | 76 | **54/154 = 35.06%** | 22 | 0 | 63.4 tok/s |
| mtp3 | zh | 96 | 48/137 = 35.04% | 46 | 1 | 49.3 tok/s |
| mtp3 | num | 88 | **55/96 = 57.29%** | 32 | 0 | 78.3 tok/s |
| plain | zh / num | 96 / 88 | — | — | — | 30.4 / 32.7 tok/s |
判读：**距离 0.9 的目标仍远**；dflash2 在自然语言 prompt 上甚至只有 4.5%（19 轮里 57 个 fallback step＝大量轮次颗粒无收），
在数字/重复性文本上 35%。MTP 35%~57% 与其历史带（43-56%）一致，**不是 0.9**。
⇒ 补丁 A 解决的是"致命停止"，**没有**解决"接受率低"。

### 3. 本轮更重要的发现：分歧不在草稿，而在 verify 路径本身
Exactness（贪心，投机理论上必须逐 token 复现 plain）：
- `dflash2 zh` 在第 32 字符分歧；`mtp3 zh` 在第 42 字符分歧；
- `dflash2 num` 在第 284 字符分歧；**`mtp3 num` 逐字相同**。
⇒ **两个不同草稿后端在同一条 verify 路径上都偏离 plain** ⇒ 不是草稿质量问题，而是 `target_verify_batch` 与 plain
解码不等价（同一模型同一上下文给出不同 argmax）。这条比"接受率低"更根本：只要 verify ≠ plain，接受率的上限就被
verify 自身的行为封住。

### 4. 已排除的一个假设（并顺带定位到另一个真 bug）
假设：verify 漏了 `apply_final_logit_policy`（契约注释写明"each lm_head logits production site applies..."）。
实测：实现是 `if constexpr (softcap > 0 || multiplier != 1)` ⇒ **对 qwen3 家族是编译期 no-op**，配置里两者皆无
⇒ 对 qwen 无影响，**不是**分歧原因。
但：**Muse 声明了 `final_logit_softcapping = 20.0` 与 `output_multiplier = 0.196`** ⇒ Muse 上投机 verify 的
argmax 会**漏掉 policy**，与 plain 不等价 ⇒ Muse + 投机解码会静默走错。**这是新发现，列入下一趟**（verify 侧补 policy）。

### 5. 本轮修掉的一个"假结论"陷阱（方法论，值得记住）
测量脚本最初没传 `--no-thinking`：回答全进 `reasoning_content`，`content` 是空串，两侧空文本会被
exactness 判成 **IDENTICAL** —— 等于用"空比对空"证明"verifier 精确"。
已修：serve 加 `--no-thinking`；并在快照里加"任一侧为空 ⇒ INVALID，不是证据"。**空结果永远不是一致性证据。**
""" )
print("M_patchA_effect.md 已追加判读")

T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 124. 构建绿灯 + 补丁 A 首批实测 + 一个更根本的发现（""" + stamp + """）
**构建**：S55 只修 3 处就把树修绿（`layouts_impl.h` 补 `#include "product/kv_options.h"`、designator 顺序交换、
`text_context_impl.h` 把 N3/S38 的 kvcalib 块搬出函数体）。证据：`ninfer` 17:42:33 / `ninfer-serve` 17:44:00，
三个 variant `.o` 均晚于 `program_impl.h` 16:31 ⇒ 补丁 A 真进了对象；两个 make rc=0 且重跑 no-op。
**补丁 A 实测**：`gen=2` 立即停消失（dflash2 zh/num = 83/76，plain 对照 96/88）；接受率 dflash2 4.51%/35.06%、
mtp3 35.04%/57.29%（详见 `_collab/M_patchA_effect.md` 判读）⇒ **致命症状解决，0.9 的目标未达**。
**新发现（比接受率更根本）**：exactness 显示 **MTP 与 dflash2 都偏离 plain**（num 上 MTP 逐字相同）
⇒ 分歧在 `target_verify_batch` 与 plain 解码**不等价**，不在草稿。已排除"verify 漏 logit policy"（对 qwen 是
编译期 no-op），但**Muse 有 softcap 20 + output_multiplier 0.196 ⇒ Muse 的投机 verify 会漏 policy，是真 bug**，列下一趟。
**方法论修正**：测量脚本没传 `--no-thinking` ⇒ content 为空、两侧空串会被判 IDENTICAL（用"空对空"证明 verifier 精确）；
已加 `--no-thinking` + "空文本判 INVALID"防呆。**空结果永远不是一致性证据。**
""" )
print("_TODO.md §124 已追加")
