#!/usr/bin/env python3
import pathlib, hashlib, datetime
J = pathlib.Path("/mnt/c/Users/User/Documents/ziqinzhang")
T = J / "_TODO.md"
old = T.read_text(errors="replace") if T.exists() else ""
block = """
## 2026-09-11 窗口①/② 状态（自动同步）

### 窗口①：FP32 partial_acc（PART-B 超集，29 文件）—— 已落地，编译中
- P7 双路对比结论：A(21文件) 与 B(29文件) 是**同一改动集**，B 为超集；**B 的关键订正**：R31 点名的
  `dense/context`（32Q/8KV）在 targets 里**无调用者**，plain(T=1) 的 live split-K 是
  `ops::gqa_attention` → `ops/kernel/gqa_attention_decode_{bf16,cuh}`（27B=24Q/4KV/D256，T≤6→SmallT，T=8→Prompt）。
- 落地形态：kernel 签名 / 打包存储（int4·pack_bf16x2 → make_float2·float2，4 个 128-bit 暂存核每 lane 仅 2 相邻 d）
  / 分配 DType::BF16→FP32 / reduce 去 __bfloat162float / `.data` 转换 31 处。
- 成本（两路一致）：live T=1/window=1200 partial 足迹 228→456 KiB/层 = 该层注意力 +19%，**整 step 权重流量 +0.2%**；
  verify 走 Prompt（T=8）该缓冲 0 字节 ⇒ 不退化。最贵项 swa/bidirectional T=16/W=4096 的 8 MiB 往返，
  已备分组回退顺序（E 组优先）。
- 判据（复测）：`verify 列 0 与 plain 一致率 69% → ≥90%`、`zh plain vs df2` 首次发散前移/消失、接受率与 tok/s 不退化。

### 窗口②：暂存 TU 拆分 —— 已就绪且与补丁后源码自洽
- **实证隐患**：staged 产物是**补丁前**抽出的 ⇒ 原样落地会把 `_partial.cuh`/`_smallt.cu` **静默退回 BF16**。
- 处置：覆盖型产物从树里幽灵副本回抄（`e12ce1a6→9c4a3ae3`、`15c1794a→f709636e`；字节账 −40/−8 精确吻合）；
  e8 兄弟 TU 施加同一替换（`_e8_arms.cuh.new` 1 处，`42ba0988→5e2a0e7c`）；**断言无 BF16 partial 残留 PASS**。
- 两张落地脚本 md5 表已同步（`_land_split.sh` 3 处、`_land_s6.sh` 2 处，FAILS=0）。
- **无冲突**：UNIFY-A 对被拆三文件命中 0；树里 live TU 为 6 处 FP32 / 0 处 BF16。

### 修② 定案关闭
`nvfp4_gdn_snapshot_plan.cpp:41-43` UNIFY-A 注释明确覆盖 projection 与 conv/store epilogue 两者，
`AllowA4` 下 T≤16 一律 `SmallTFusedA16` ⇒ 无 gemv 路由可切，3→16 是**代码级可证的 no-op**（零影响有解释，无需日志行）。
"""
stamp = datetime.datetime.now().strftime("%F %H:%M")
T.write_text(old + block)
print(f"_TODO.md: {len(old.splitlines())} -> {len(T.read_text().splitlines())} lines  (synced {stamp})")
