#!/usr/bin/env python3
"""记录：S54 的裁决 + 导入器第二轮修正 + 上一条 CLEAR 的更正。"""
import datetime
import pathlib

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# 1) 把裁决追加到 S54 计划文档末尾（后续执行者按它做）
plan = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_collab/S54_spark_target_plan.md")
plan.open("a", encoding="utf-8").write("""

---

## §7 协调方（M）裁决 — """ + stamp + """

1. **target id**：C++ 标识符统一走 `cpp_ident()` ⇒ `spark_x2_5_4b`（小数点/连字符一律折成下划线，
   已在 `tools/archkit/adapt.py` 落地并验证：生成物第 4 行是 `namespace ninfer::targets::spark_x2_5_4b::detail`）。
   对外 model-id 仍保留 `spark-x2.5-4b`（HF 侧名字），**不要再造第二套命名**。
2. **首版权重档 = BF16**。理由：Spark 无任何量化产物、且新架构没有几何注册，NVFP4/FP8 都要先有自己的量化与
   几何校验；先用 BF16 把**数值对齐**做成（与 `modeling_spark.py` 逐层比），量化作为后续性能项。
3. **SWA 首版范围 = 上下文 ≤ 512 token**。理由（实证）：`W=512` 时滑窗注意力在语义上等价于全注意力，
   而 36 层全量 KV = 144 KB/token（32k ≈ 4.7 GB），主 KV 池又没有 per-layer 容量，SWA 的省显存收益
   不会自动出现；同时 BF16 解码核的 `window` 语义是"可见键数"、`sliding_window_tokens` 从不被赋值、
   草稿侧 `ops::swa` 是 D128/32/8/W4096 固定域不可复用。⇒ **v1 验收 = 短上下文（≤512）与 HF 对齐**；
   长上下文必须等 SWA 通路（per-layer 窗口表 + 主通路赋值点 + BF16 核语义）真正落地。
""")
print("S54 计划已追加裁决 §7")

# 2) _TODO.md §123
T = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang/_TODO.md")
T.open("a", encoding="utf-8").write("""
### 123. S54 推翻乐观结论 + 导入器第二轮修正 + 裁决（""" + stamp + """）
**更正我上一条**：我上轮根据 manifest 的 `hook 7 / new_op 0` 说 Spark"VERDICT: CLEAR"——**错了**，
那是分类器不全造成的。S54 的只读分析给出 5 处硬发现，我逐条到源码核实（不是采信）：
1. **生成的 config.h 编不过**：`namespace ninfer::targets::spark_x2.5_4b::detail` 带小数点
   （生成器只替 `-` 没替 `.`）⇒ 阻塞级。**已修**：新增 `cpp_ident()`，生成物第 4 行现在是 `spark_x2_5_4b`。
2. **主注意力 16Q/4KV@head_dim256 未注册**：引擎注册表只有 `16/2@256, 24/4@256, 32/2@128`；
   `wrapper/gqa_attention.cpp:25-30` 按 q_heads 反推（16→2）再抛错。**已加探测器**：直接解析
   `src/ops/kernel/gqa_attention_geometry.cuh` 的 `GqaGeometry<...>` 别名表比对 (q,kv,hd)，
   未注册即 new_op（报告里会列出注册表现有项）。
3. **逐头输出门是 new_op 不是 hook**：`sigmoid_mul` 要求四维同形（`wrapper/sigmoid_mul.cpp:36-40`），
   HF 的 `[n_q,T]` 逐头广播没有算子。**已改判**并写清零算子出路（把 g_proj 展开成
   `[n_q*head_dim, hidden]` 的元素级门控，BF16 约 +755 MB）。
4. **gated GELU MLP 也是 new_op**：`ops::gelu` 是 in-place 单元激活，`gelu(gate)*up` 的两输入乘全树无算子
   （ops 下只有 silu_mul/sigmoid_mul/gelu/causal_conv1d_silu）。**已改判**。
5. **qk-norm 缺失**：家族无条件 `rmsnorm(q,k)`（`text_context_impl.h:958-959`）而 Spark 没有 qk-norm
   ⇒ **已加探测器** `attn:qk_norm=absent`（hook：需 `qk_norm_enabled()` 门）。
6. 好消息（S54 实证）：`ops::rope` 语义与 HF **完全一致**（前 R 维 + split-half + θ^(-2i/R)），
   R=256 撞上 `kRopeMaxHalf=128` 边界但可用 ⇒ **rope 不需要新内核**，只是两种层型都落通用核（性能项）。
**重跑导入器后的诚实结论**：`[blocked] config.h withheld as config.h.BLOCKED (unresolved:
attn:headwise_output_gate(sigmoid); mlp:act=gelu(gated); attn:head_geometry(16q/4kv@256))`，
hook 8 项（含 qk_norm=absent）、covered 1 项（tied）、post 1 项。**Spark 在几何/算子到位前不该被转换**。
**裁决**（已写入 `_collab/S54_spark_target_plan.md` §7）：id 用 `spark_x2_5_4b`；首版权重档 **BF16**；
SWA 首版范围 = **上下文 ≤ 512 token**（W=512 时滑窗≡全注意力），长上下文必须等 SWA 通路。
""" )
print("_TODO.md §123 已追加")
