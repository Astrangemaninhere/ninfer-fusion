# UNIFY-A：统一小 T 家族（T∈[1,16]）的算术 —— 分派点清单 + 统一 diff + 性能代价（2026-09-11）

树：`/home/user/ninfer-fusion`（只读分析；**未改构建树、未编译、未跑 GPU**）。
产物：本报告 + `_collab/UNIFY_A_patch.diff`（`patch -p1 --dry-run` 已通过，见 §7）。
本文所有 `file:line` 均为**改动前**（当前树）行号。构建树 `/home/user/ninfer-fusion` ↔ 报告/diff 落盘在 Windows 侧 `_collab/`。

---

## 0. 结论先行

1. **统一方向被实证钉死，而且是"零 verify 代价"的方向**：dflash2 verify 的 T=W（实测 W=8）在**每一处**分派点上都已经站在"家族"那一侧；plain decode（T=1）才是那个被单独优待的形状（gemv decode / A16 档）。
   ⇒ 本次统一**只把 T=1（及少数 T=2..7 的档位阈值）搬到 verify 已经用的那条路上**，**不动 verify 的任何 route**（唯一例外见 §2 末：lm_head 在 W=9..16 的 split-K 宽度 8→16，为覆盖 W=16）。
   ⇒ **verify 侧 tok/s 按构造不可能退化**（同一 kernel、同一 schedule、同一 grid 形状），T=1 侧代价见 §5。
2. 覆盖到的 T 分派点：**14 处**（attention 输入投影 ×2 档、核心注意力 ×1、GDN 输入/conv/gating ×4、MLP gate_up/down ×4、lm_head ×1、bf16 残差 ×1，另加 3 个族群常数/断言放宽）。diff 共 **17 文件 28 处**。
3. **有一处无法在"不牺牲性能"的前提下统一**：**核心注意力**（T≤6 走 `SmallT`、T∈[7,16] 走 `Prompt`）。T=1 若改走 `Prompt`，一层注意力只剩 `24` 个 CTA（27B 24Q/4KV）⇒ 长上下文下单 CTA 扫全部 visible keys，延迟爆炸；反向（让 W=8/16 走 SmallT）必须按 6 token 切块 ⇒ 2~3 次串行 launch 且仍是不同 split 数。**建议只统一到"同精度档"**（见 §6）。
4. 性能代价（T=1，27B，batch=1，5090，乐观 1.7 TB/s）：**估算 +1.5%~3.5%/decode step**（唯一来源是新增的"激活量化"与"split-K reduce"小 kernel 节点，约 60~120 个 × 1.5~2 µs；**访存量级不变**）。**必须实测**，若 >3% 按 §5 的粒度回退。
5. 量化判据：**逐位同算术的可达性取决于"该家族的 K 归约轴是否随 ActiveTokens 变"**。nvfp4 small-T 家族在 T∈[1,16] 上**所有 K 归约轴都与 ActiveTokens 无关**（§4.1 逐表达式对照）⇒ 逐位同算术；bf16 small-T 家族**不是**（`kValuesPerLane`/`kPhaseOrder` 随 ActiveTokens 变）⇒ bf16 的 attn_input 用小 T 核**无法**逐位统一，故未纳入（§6）。

---

## 1. 证据基础（本报告只依赖代码事实 + 已归档实测）

- 调用链（27B + dflash2）：`text_context_impl.h:810/816`（**plain decode = `ordinary_decode_batch`，width 绑定为 1，phase = `Phase::Verify`**）与 `text_context_impl.h:867/873`（verify：width=W，同一 phase、同一 `run_layers`）。
  ⇒ **plain 与 verify 是同一段代码、同一 phase**，唯一差别是 `x.ne[1]`（与 width/batch 元数据）⇒ 所有 T 分派点同时被两者经过。这是"统一 T 家族即统一 spec 与 plain"的结构性依据。
- 权重档位（决定哪些 dispatcher 真的在路径上）：`src/targets/qwen3_6_27b/impl/load/bindings.cpp:341-390`（`bind_qwen38_nvfp4_text_layers`，Qwen38Nvfp4Dspark/DFlash2 用）：attn in = **FP8**、o_proj = **FP8**、gdn in/out = **FP8**、gdn a_b = **BF16**、mlp gate_up/down = `layer<56` **NVFP4** / `layer>=56` **FP8**、lm_head = **FP8**（`bindings.cpp:422,445` + `variant.cpp:35-48`）。
  `bind_nvfp4_text_layers`（`bindings.cpp:271-338`，Qwen36Nvfp4 用）另有：attn in = NVFP4（`layer∈{3,7,11,15,19,23}` 为 BF16）、o_proj = NVFP4（`layer∈{3,7}` 为 BF16）、gdn out（`layer==4` 为 BF16）、gdn in = NVFP4。
  ⇒ 两套 profile 都覆盖，diff 对两套都生效。
- R24 已证伪的结论不再复用：修④（TMA alpha）是假阳性且已回退；chunk 依赖（3.2%）是舍入级，不属本任务。
- **一条修正 S_E §2.8 的事实**：lm_head 不是"恒 T=1"。verify 走 `text_context_impl.h:877-881`（`flat_logits = logits.view({vocab, width*batch})` → `ops::linear(flat_hidden, *lm_head_, flat_logits)`），即 **verify 的 lm_head 是 T=W**；只有 plain 是 T=1。它因此是**真正的 T 分派点**（见 §2 表 15）。

---

## 2. 逐分派点：谓词 → 现状 → 统一方案（★ = plain(T=1) 与 verify(T=W=8) 之间确实分叉）

| # | 分派点（file:line） | 分派谓词原文 | T=1 现状 | T=8 现状（verify） | 统一后（本 diff） | 是否逐位同算术 |
|---|---|---|---|---|---|---|
| 1 ★ | `gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:42` | `if (tokens == 1) { return DecodeFusedA16; }` / `:43 if (tokens <= 16) { return SmallTFusedA16; }` | `nvfp4_gdn_snapshot_decode_launch` = `nvfp4_gemv_kernel`（4 条累加链） | `nvfp4_gdn_snapshot_small_t_launch` = `nvfp4_small_t_kernel`（1 条链） | 删除 `tokens==1` 分支（`@42`，A16Only 同类分支 `@38` 一并删） | ✅ 同核同序（§4.1、§4.2） |
| 2 | `gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:41-45` | `if (active == 1) { …decode… } else { …small_t… }` | decode gemv | small_t | 恒 small_t | ✅（仅 A16Only 可达，仍统一） |
| 3 ★ | `gdn_input_proj/fp8/fp8_gdn_conv_fused.cu:117-121` | `if (x.ne[1] == 1) { launch_snapshot_decode(...); return; }` | `fp8_gemv_kernel`（gemv decode） | `fp8_small_t_kernel` + `GdnConvOutput<8,…>` | 删除该分支 → 走 `kSnapshotLaunchers[0]` | ✅（同核；conv 逐 token 独立，§4.2） |
| 4 | `gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:56-60` | `if (active == 1) { …decode… } else { …small_t… }` | decode gemv | small_t | 恒 small_t | ✅ |
| 5 ★ | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:33,37-46` | `if (tokens == 1 && getenv("NINFER_T1_W4A4")==nullptr) → DecodeFusedA16; if (tokens <= 4 && …) → SmallTFusedA16; … if (tokens <= 48) → FusedW4A4;` | **A16 档**（gemv decode；T2..4 → A16 small_t） | **W4A4 档**（`M48N64` MMA 融合） | AllowA4：`tokens <= 48 → FusedW4A4`（删 T≤4 的 A16 分支与 env 开关）；A16Only：`tokens <= 16 → SmallTFusedA16`（删 T==1 分支） | ✅ MMA 家族单累加器、M 无关（§4.3） |
| 6 | `linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:25` | `return tokens == 1 \|\| tokens >= 3 ? A8 : A16;` | A8 | A8（但 **T=2 是 A16**） | 恒 A8 + planner `interval_uses_a8` 恒真 | ✅（同 A8 schedule，M 掩码差异为零填充） |
| 7 ★ | `attn_input_proj/fp8/fp8_attn_input_plan.cpp:48-54` | `if (active == 1) { …decode… } else { …small_t… }` | A16 → decode gemv | A16（`tokens>=11` 才是 A8）→ small_t | 恒 small_t（档位不动） | ✅ |
| 8 ★ | `attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:24,49-55` | `return tokens >= 4 ? W4A4 : A16;` | **A16 档**（decode gemv） | **W4A4 档**（M32N64） | 恒 W4A4 + A16 分支恒 small_t | ✅（§4.3） |
| 9 | `linear_add/fp8/fp8_linear_add_plan.cpp:43-47` | `if (active == 1) { …decode… } else { …small_t… }` | decode gemv | small_t（`first_a8 = 22/25` ⇒ T=1 与 T=8/16 同档） | 恒 small_t | ✅ |
| 10 ★ | `linear_add/nvfp4/nvfp4_linear_add_plan.cpp:26-27,40-44` | `first_w4a4 = input_rows == 6144 ? 7 : 8; return tokens >= first_w4a4 ? W4A4 : A16;` | **A16 档**（decode gemv；o_proj K=6144、down K=17408） | **W4A4 档**（M32N64） | 恒 W4A4 + A16 分支恒 small_t | ✅ |
| 11 ★ | `linear/nvfp4/nvfp4_dispatch.cpp:32-39` | `AttnInput: tokens>=4` / `MlpGateUp: tokens>=5` / `Residual*: tokens>=8` `? W4A4 : A16` | **A16 档** | **W4A4 档** | 三个 problem 恒 W4A4；`:60-64` 恒 small_t | ✅ |
| 12 ★ | `gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:30-31` | `{{1,1}, GemvPairedRows}, {{2,8}, SmallTSplit10}` | `GemvPairedRows`（split_k = **1**，1 kernel） | `SmallTSplit10`（split_k = **10**，partial+reduce 2 kernel） | `{{1,8}, SmallTSplit10}`（表 6→5 项）+ 合法性 `cols>=1 && cols<=8` | ✅ 同核同 split 数 |
| 13 | `linear_add/bf16/bf16_linear_add_plan.cpp:17-19` | `if (tokens==1) → Decode; if (tokens<=4) → SmallT; if (tokens<=48) → AggregateMma;` | Decode gemv | `AggregateMma`（`UpTo32 = Bf16MmaSchedule<32,32,256,16,8,…>`，固定 schedule，T 无关） | `if (tokens <= 48) → AggregateMma` | ✅ 固定 schedule，M 无关 |
| 14 ★ | `linear/fp8/fp8_config.h:140-157`（`Fp8VocabularyA16SmallTMmaProductionSchedule`）+ `:155` | `kKWarps = ActiveTokens <= 8 ? 16 : (ActiveTokens <= 24 ? 8 : 4);` | `kKWarps=16`（split-K 16） | T=8 → 16（**一致**）；**T=9..16 → 8（不一致）** | `kKWarps = ActiveTokens <= 16 ? 16 : …` | ✅ 固定 split-K 宽度 |
| 15 | `linear/fp8/fp8_config.h:304` + 家族常数；`fp8_small_t.cuh:60`；`linear/nvfp4/nvfp4_config.h:196`；`nvfp4_small_t.cuh:215` | `kFp8FirstSmallT = 2` / `kNvfp4FirstSmallT = 2` / `static_assert(ActiveTokens >= 2)` | 家族表不含 T=1 ⇒ 只能走 decode | 家族表含 T=8 | 常数→1、断言→`>= 1`（各 launcher 表自动扩展为 1..N） | — 基础使能项 |
| 16 | `linear/fp8/fp8_config.h:397`（`Fp8LinearSmallTProductionSchedule<Fp8GdnInputGeometry,T>`） | `kValuesPerLane = ActiveTokens >= 5 && ActiveTokens <= 6 ? 8 : 16;` | 16 | 16（**但 T=5,6 是 8**） | 恒 16 | — 把家族内**算术轴**钉死（§4.1） |

**verify 侧唯一被改动的 route**：#14 的 **W=9..16**（lm_head split-K 8→16）与 #5/#6/#8/#10/#11 在 **W=2/4** 上的档位（因为那些阈值把 T=2..4 划给了 A16 侧，而统一把它们拉到 verify 侧）。**W=8（实测 verify 宽度）在所有 16 项上 route 完全不变。**

注（lm_head 的另一条 T 分档，在本家族之外）：`fp8_dispatch.cpp:65` 的 `problem == Vocabulary && x.ne[1] >= kFp8VocabularyFirstA16GemmT(42)` 会切到 `launch_fp8_vocabulary_a16_gemm`；T≤41（含 T=1..16 全体）恒走 `launch_fp8_vocabulary_a16_small_t`（分块常量 `kFp8VocabularyLastA16SmallTMmaT = 48` ⇒ T≤48 不切块）。故 [1,16] 内 lm_head 只有 #14 一个算术轴。

其它被检查而**未改**的 T 分派点（理由见 §6）。

---

## 3. 统一 diff

`_collab/UNIFY_A_patch.diff`（17 文件 / 28 处 / 373 行）。文件清单：

```
attn_input_proj/fp8/fp8_attn_input_plan.cpp          attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp   gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
gdn_input_proj/fp8/fp8_gdn_input_plan.cpp            gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp     linear/fp8/fp8_config.h
linear/fp8/fp8_small_t.cuh                           linear/nvfp4/nvfp4_config.h
linear/nvfp4/nvfp4_dispatch.cpp                      linear/nvfp4/nvfp4_small_t.cuh
linear_add/bf16/bf16_linear_add_plan.cpp             linear_add/fp8/fp8_linear_add_plan.cpp
linear_add/nvfp4/nvfp4_linear_add_plan.cpp           linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
```

- 生成方式：读原文件 → 内存内做**精确原文替换**（每处断言"命中且仅命中 1 次"，否则脚本失败退出）→ 与原文件做 diff。**构建树全程只读**（脚本见 `%TEMP%\ua_patch.py`；CRLF 文件 4 个的 CR 已保留在 hunk 中）。
- 与 route 配套的 **workspace planner 改动**（不改就运行期抛异常）：
  - `nvfp4_linear_swiglu_plan.cpp:100` `if (policy == A16Only || max_tokens <= 4) return 0;` → 去掉 `max_tokens <= 4`（T≤4 现在要 W4A4 激活量化 workspace）；`:103` 的 `max_tokens >= 5` → 去掉。
  - `fp8_linear_swiglu_plan.cpp:56-57` `interval_uses_a8 = min_tokens==1 || max_tokens>=3` → 恒 `policy == AllowA8`。
  - `nvfp4_linear_add_plan.cpp:58` / `nvfp4_attn_input_plan.cpp:67` / `nvfp4_dispatch.cpp:77` / `fp8_attn_input_plan.cpp:66` 的 `== W4A4/A8 ? 容量 : 0` **自动**跟随 route 变化（无需改）。`bf16_gdn_gating` 的容量由 `route_capacity()` 在每个 route 的**区间端点**取最大（`bf16_gdn_gating_proj_plan.cpp:276-288`）⇒ `cols=1` 新 route 的 `10*1*96*4 = 3840 B` partial 自动被计入。

---

## 4. 逐位同算术的**逐表达式论证**

### 4.1 nvfp4 small-T 家族：K 归约轴全部与 ActiveTokens 无关（T∈[1,16]）

`nvfp4_small_t.cuh:62-200`（`compute_nvfp4_small_t_rows`）里决定"每个 (row,token) 元素的 fp32 累加序列"的量，逐个查它是否随 ActiveTokens 变：

| 量 | 表达式 | GdnInput/通用 specialization 在 T∈[1,16] | 影响算术？ |
|---|---|---|---|
| `kValuesPerPhase` | `32 * kValuesPerLane`（`:62`） | `kValuesPerLane = (T∈[17,20]? 8 : 16)` ⇒ **恒 16** | 是（决定 phase 数） → **恒定** |
| `kPhases` | `kInputRows / kValuesPerPhase`（`:64`） | 10 | 恒定 |
| `warp_phase` | `phase * kWarpsPerRow + warp_in_row`（`:91`） | `kWarpsPerRow = 1` ⇒ `= phase` | 是 → **恒定** |
| `kPairsPerLane` | `kValuesPerLane/2`（`nvfp4_config.h:71/105`） | 8 | 是 → 恒定 |
| 累加链下标 | `(2*pair) & (kAccumulatorChains-1)`（`:142/145`） | GdnInput spec `kAccumulatorChains = 1` ⇒ 恒 **0**（单条顺序链） | 是 → 恒定 |
| 激活下标 | `phase*(kValuesPerPhase/2) + warp_in_row*(kValuesPerWarpPhase/2) + lane*kPairsPerLane + pair`（`:166-168`, `:113-115`） | 不含 T | 是 → 恒定 |
| 权重 code 偏移 | `parent_row*kCodeBytesPerRow + phase*(kValuesPerPhase/2) + warp_in_row*(…) + lane*kPairsPerLane`（`:99-102`） | 不含 T | 恒定 |
| 收尾 | `total = Σ_chains`（1 项）→ `warp_reduce_sum(total)`（`:273-276`） | 32 lane × 16 value 的 lane↔value 映射由 `kValuesPerLane` 定，恒定 | 是 → 恒定 |
| `token0` / grid / `kTokenTiles` | `:222-233` | 只决定"哪块 CTA 拥有哪些 token" | **否** |
| `kWarpsPerCta`（8/16/4） | `nvfp4_config.h:206/223/240/254` | 只改 `kRowsPerCta` = 行↔warp 映射 | **否** |
| `kActivationAccess`（SharedPhase vs TokenPacked） | 只改激活**搬运路径**（shared 中转 vs 直读 global） | 位模式相同 | **否** |

⇒ 对同一个 token，无论它落在 T=1 还是 T=8 的调用里，**FMA 序列逐条相同**，收尾归约树相同 ⇒ **逐位一致**。
**这也是"修② 单变量测不到效果"的原因**：修② 只把阈值从 3 放宽到 16，T=1 仍走 gemv（4 链）；必须像本 diff 一样把 T=1 也送进 small-T 核才动到算术。

### 4.2 GDN conv/state epilogue：逐 token 独立（两条路共用同一 epilogue）

`gdn_input_proj/gdn_conv.cuh:65-120`（`GdnConvEpilogue::store<Tokens>`）：
- 每行独立；`for (token = 0; token < Tokens; ++token)` **顺序**执行，`s0/s1/s2` 依次前移，`s2 = bf16(p)`（`:118`，修① 已落）；
- 同一行的 token t 只依赖本行 token < t 与入口 state（`:76-82` 从 `state_read[initial_base]` 读）。
⇒ 用 `Tokens=1` 调一次 与 用 `Tokens=W` 调一次，**token 0 的算出值逐位相同**（其余 token 同理，只是分次调用）。
`GdnConvOutput<Tokens,Publish>::store_row`（`gdn_conv_output.cuh:27-38`）把 q/k/v 走 `conv.store`、z 走 `bf16(projected)`；gemv 路的 `store(row,0,x)`（`:40-46`）**内部就是 `store_row(row,{x})`**（`static_assert(Tokens==1)`）⇒ 两条路的 epilogue 是同一份代码。

### 4.3 nvfp4/fp8 MMA 家族：单累加器、M 无关（S_E §2.1 结论，本地复核）

- `nvfp4_w4a4_mma.cuh:234` `float accumulators[kMmaM][kMmaN][4] = {}` —— **单链、无 split-K、无 atomic、无 partial**；
  `:248 for (k_tile …)` → `:254 for (local_k64 …)` → `:308 mma_nvfp4_e4m3(...)`：K 顺序与 M（token 块）无关；
- 越界 token 行零填充（`:89/:104/:122 valid` 判断）、store 处 `:373 if (token < tokens)` 拦截 ⇒ 行填充**不污染**（加 0.0F 在 fp32 累加下是精确的）；
- 同一 problem 的 swiglu/attn/residual 各自**只有一个** `launch<M…>` 实例覆盖 `tokens <= 64`（`nvfp4_linear_swiglu_w4a4.cu:94 launch<M48N64>`、`nvfp4_attn_input_w4a4.cu:96 launch<M32N64>`、`nvfp4_linear_add_w4a4.cu:41 launch<M32N64>`、`nvfp4_dispatch.cpp` → `nvfp4_w4a4.cu:72 launch<M32N64>`）⇒ T=1 与 T=8 是**同一个模板实例**。
- fp8 A8 家族：`Fp8LinearA8ProductionSchedule<Geometry>` 只按 Geometry 特化（`fp8_a8_schedule.cuh:11-39`），`tokens % kBlockTokens == 0` 只切 `FullTokens` 掩码（`fp8_a8.cu:107-111`）⇒ 同一 kernel。
- fp8 vocabulary MMA：唯一的**算术轴**是 `kKWarps`（split-K 宽度），已钉死（§2 表 14）。

### 4.4 bf16 small-T 家族：**不是**逐位统一的（因此未纳入）

`linear/bf16/bf16_config.h:126-156`：`kValuesPerLane = (K=6144 且 T∈{3,8}) ? 16 : 8`、`kPhaseOrder = kSequential ? Sequential : RowSwizzled`（`kSequential = T<=9 || T>=17`）——**两条都是算术轴**（lane↔value 映射、phase 访问顺序）。所以"让 bf16 attn_input 的 T=1 改走 small_t"只会把 T=1 换到另一个**同样与 T=8 不同**的算术上，属**假修复**，故 `attn_input_proj/bf16/bf16_attn_input_plan.cpp:7-13` **不在 diff 内**（§6）。
bf16 **linear_add** 例外：其 `AggregateMma` 路是**固定** `Bf16MmaSchedule<32,32,256,16,8,3,1,…>`（`bf16_linear_add_gemm_mma.cu:66-71`，`UpTo32` 覆盖 T≤32）⇒ 无 ActiveTokens 依赖 ⇒ 已纳入（表 13）。

---

## 5. 性能代价（roofline/访存量级）

**访存量级不变**是这次改动的关键性质：统一只改"谁算"，不改"读多少权重"。

27B、batch=1、5090（按可达 1.7 TB/s 计）单 token 权重流量：

| 算子 | NVFP4 | FP8 | bf16 |
|---|---|---|---|
| attn input (14336×5120) | 36.7+4.6 = 41 MB (24 µs) | 73 MB (43 µs) | — |
| gdn input (16384×5120) | 41.9+5.2 = 47 MB (28 µs) | 84 MB (49 µs) | — |
| o_proj / gdn out (5120×6144) | 15.7+2.0 = 18 MB (10 µs) | 31.5 MB (19 µs) | 63 MB (37 µs) |
| mlp gate_up (34816×5120) | 89.1+11.1 = 100 MB (59 µs) | 178 MB (105 µs) | — |
| mlp down (5120×17408) | 44.6+5.6 = 50 MB (29 µs) | 89 MB (52 µs) | — |
| lm_head (248320×5120) | — | **1.27 GB (≈750 µs)** | — |
| gating a_b (96×5120) | — | — | 0.98 MB (0.6 µs) |

整步（16 全注意力层 + 48 GDN 层 + lm_head，layer<56 NVFP4 MLP / layer≥56 FP8 MLP）≈ **9.9 GB + 1.27 GB ≈ 6.6 ms**，与实测 ~9 ms/step（111 tok/s）相符 ⇒ **步长是带宽受限**，因此：

| 改动 | T=1 代价 | 机制 | 量级 |
|---|---|---|---|
| 1,3,4 GDN conv/输入 → small_t | ≈0 | 同字节、同总线程数（1024 CTA×256t ↔ 2048×128t）；1 链 vs 4 链的依赖深度由 warp 并行掩盖 | ±1% |
| 7,9,10,11,2 T=1 → small_t（A16 档内） | ≈0 | 同档同核，只换 schedule | ≈0 |
| 5,8,10,11 档位 A16→W4A4（T≤3/4/6/7） | **+1 个"激活量化"kernel 节点/次** | 量化访存 = 读 `5120×2 B`、写 ~`2.7 KB`，本身 <2 µs；代价是**节点延迟**（CUDA graph 上 ~1.5~2 µs） | NVFP4 profile：gate_up 56 层 + down 64 层 + attn in 16 层 ≈ **+0.27 ms** |
| 12 gating T=1 → SmallTSplit10 | **+1 个 reduce kernel 节点/层**（`bf16_gdn_gating_proj_kernels.cu:363-380` 两次 launch） | partial 缓冲 `10×1×96×4 = 3840 B`；reduce 为 96 float | 48 层 × ~1.5 µs ≈ **+0.07 ms** |
| 13 bf16 linear_add T≤4 → AggregateMma | ≈0 | M32 tile 对 1 token ⇒ 32× 空算（1.0 GMAC，tensor core 上 <2 µs），权重流量不变（63 MB/37 µs，带宽受限） | ±1% |
| 14 lm_head kKWarps 8→16（W=9..16） | — | CTAs 翻倍、并行度更高，权重流量相同（1.27 GB） | verify 侧 ≈0 |
| **合计（T=1）** | | | **≈ +0.34 ms ≈ +1.5%~3.5%**（最坏，若节点完全不能重叠） |

**verify（W=8）代价 = 0**：所有 route 保持原样（§2 末）。`_ga_check` 的 plain 侧若因此慢 >3%，按此粒度回退：先回退 #12（gating，代价最小收益也最小）→ 再回退 #5（swiglu，代价最大）→ 最后回退 #8/#10/#11 的档位（这几项是"精度档"统一的唯一手段，回退它们等于放弃 W∈{2,4} 与 T=1 的档位一致）。
**真正的零代价替代（若实测超预算）**：把激活量化**融进上游 epilogue**（gate_up 的上游是 rmsnorm/residual_add 的 bf16 输出），即可在不改 route 的前提下消掉那 ~0.27 ms；这需要额外一轮改动，不在本 diff 内。

---

## 6. 无法统一者（明确写出 + 替代）

### 6.1 核心注意力：T≤6 `SmallT` ↔ T∈[7,16] `Prompt`（**性能不可兼得**）

- 谓词：`wrapper/gqa_attention.cpp:404-412`
  ```
  if (width >= 1 && width <= kSmallTChunkTokens /*6*/) { return SmallT; }
  if (batch_size > 1) { return ChunkedSmallT; }
  const prompt_visible_keys = width <= 12 ? 512 : 1024;
  if (q_heads == 16 && width <= 16 && envelope.max_visible_keys > prompt_visible_keys) { return ChunkedSmallT; }
  return Prompt;
  ```
  27B 是 **24 Q-heads/4 KV**（`config.h:37-38`），故 `width ∈ [7,16]` **恒 `Prompt`**；`width ∈ [1,6]` 恒 `SmallT`。⇒ plain T=1 = `SmallT`，verify W=8 = `Prompt`（`launcher/gqa_attention_decode.cu:481 gqa_attention_uses_small_t = 1..6`）。
- 两条路的算术本质不同：`SmallT` = 沿 **key 轴 split-K**（`splits = gqa_small_t_launch_capacity(...)`，partial `acc` 为 **BF16** + fp32 `m/l`，再 combine：`wrapper/gqa_attention.cpp:330-346`），`Prompt` = 单遍 online softmax、**grid 无 key 轴**（`prompt.cu:49 div_up(tokens,kCausalPromptBr) × QHeads`）。
- 为什么不统一：
  - T=1 → `Prompt`：一层注意力只有 `24` 个 CTA；长上下文（visible keys 可达 262144）下每 CTA 串行扫完全部 key ⇒ 从"Split-K 并行"退化为"单点长链"，延迟灾难（违反"禁止串行化"）。
  - W=8/16 → `SmallT`：`SmallT` 只注册 `width ≤ 6`，必须按 6 切块（`for_each_small_t_chunk`，`wrapper/gqa_attention.cpp:349-364`）⇒ W=16 需 3 次串行 launch，且每块的 `splits` 由块宽决定（`gqa_small_t_split_count` 只依赖 window 与 kv_dtype，NT4 分支不依赖 tokens —— `gqa_attention_decode_partial.cuh:88-121`）⇒ 即使这样也**仍不是同一 split 数**。
- **替代（建议采纳）**：只统一到"**同精度档**"——两条路都已经：同一 KV cache dtype（NVFP4）、同一 `scale`、同一 Q/K 归一化与同一 fp32 `m/l` 软件；差异只在 split-K 归约树。建议把 G-A 判据在注意力这一项上记为"**同精度档 + 首次偏离由注意力贡献的部分已量化**"。
- **可离线/零改码的量化实验（给主代理）**：用 `--spec` 改 draft 宽度把 verify 宽度压到 4（`SmallT`）与 8（`Prompt`），其余全同；对比 `_ga_check.sh` 的首次偏离位置与 argmax 翻转率。若 W=4（与 T=1 同族）的翻转率显著低于 W=8，则注意力确实贡献了可观份额；若两者接近，则注意力不是主要残差，统一它的优先级可以降到最低（也免于为它付性能）。

### 6.2 其它已检查、**未**纳入的分派点（附理由）

| 位置 | 谓词 | 为何不纳入 |
|---|---|---|
| `attn_input_proj/bf16/bf16_attn_input_plan.cpp:7-13` + `plan.h:13-15` | `x.ne[1]==1 → decode; <= 22 → small_t; else mma` | bf16 small-T 的 `kValuesPerLane`/`kPhaseOrder` 随 ActiveTokens 变（§4.4）⇒ 改 route 是假修复；要真统一必须钉死这两轴并重测 T=2..22 的实测最优点 ⇒ **需用户决策 + GPU 窗口** |
| `linear/bf16/bf16_dispatch.cpp:30-33` | `t == 1 → decode; <= small_t_end → small_t` | 只对 `legacy_problem`（14336×5120 / 5120×6144）可达；该 profile 的这两个形状走 `attn_input_proj` / `linear_add`，不经 `ops::linear` ⇒ 非路径；且同 §4.4 的 bf16 问题 |
| `launcher/causal_conv1d.cu:264/274/284` | `B==1 && T==1` → decode；`T<=32` → smallt | **只被 prefill 分支调用**（`text_context_impl.h:1135` 在 `else`（非 Verify）里）；decode/verify 的 conv 走 GDN fused/snapshot 路 ⇒ 不影响 plain vs verify |
| `linear_attention/gated_delta_net/gated_delta_net.cpp:248-289` | `T_full = (T/64)*64`，`T_full>0` chunked / `T_full==0` recurrent | T≤16 时两边都是 `T_full==0` ⇒ 同走 `launch_recurrent_batch_update`（`gated_delta_net.cpp:227-239`）；chunk 分档只在 prefill（T≥64）触发 |
| `linear/fp8/fp8_dispatch.cpp:42/44/46/58`（Residual/MlpGateUp/AttnInput/GdnInput 档位） | `>= 12 / 11 / ==1\|\|>=5 / >=25` | 只在 `ops::linear` 被这些形状调用时生效；27B DFlash2 的 FP8 只用于 lm_head（Vocabulary→恒 A16，`fp8_dispatch.cpp:32-35`）⇒ 该文件的档位在路径上**不可达**（其 Residual 档位由 `linear_add` 自己的 plan 承担，已在 #10 统一） |
| `gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp:45` | `fused = width <= 3 \|\| (width >= 7 && width <= 10)` | W∈{4,5,6} → `MaterializedA16`（与 T=1 的 `FusedA16` 不同族）。**实测宽度 W=8 与 T=1 都是 `FusedA16`**（`7<=8<=10`）⇒ 真的 plain-vs-verify 对已统一；改 `width<=10` 会让 T=4..6 的实测最优点被覆盖 ⇒ 列为"可选项，需实测" |
| `sparse_moe/prefill/sparse_moe_prefill_kernels.cu:1153` | `tokens >= 47 && <= 51/52` | 35B-A3B MoE；27B dense 不涉及（S_E §1.4） |
| `kv_cache/append/launch.cu:24/57`、`launcher/gqa_attention_prefill*.cu:261/50` | `tokens >= 128 && KVHeads == 2` | 27B 是 4 KV ⇒ 恒不命中（S_E §2.6） |
| `launcher/rope.cu:85` | DFlash `tokens <= 16` | 只改 block/grid（S_E §2.4，逐位不变） |
| W8/Q5/Q6 各 `*_plan.cpp` 的精确 T 区间表 | 如 `w8_linear_add_plan.cpp:29-35` | 属于 **MTP/draft 段**（W8）与其它权重档 profile，不在"target plain vs target verify"的比较里 |
| `wrapper/gqa_attention.cpp:405`（`batch>1 → ChunkedSmallT`） | | batch=1 的 plain/verify 对不涉及；同 §6.1 |

---

## 7. 残余风险

1. **未编译**（硬约束）。风险按序：
   a. **T=1 实例化是否都能编过**：已逐条核对的关键点 —— `nvfp4_small_t_kernel` 的 `static_assert(kTokenTile <= ActiveTokens)`(=1✅)、`kRowsPerCta % 4 == 0`(=16✅)、`128 % kRowsPerCta == 0`✅；`Nvfp4SmallTSharedStorage` 在 TokenTile=1 下 `kActivationElements = 1×512`（SharedPhase）或 8（TokenPacked）✅；`Nvfp4LinearSmallTProductionSchedule` 四个 specialization 的 `kWarpsPerCta/kValuesPerLane` 在 T=1 下分别为 8/4/4/4 + 16✅；`Fp8SmallTSchedule` 六个 specialization 的 `kValuesPerLane` 在 T=1 下 16（其中 MlpGateUp 的 `kActivationAccess` 变 `SharedPhase`）✅；`fp8_linear_add_small_t.cu`/`fp8_attn_input_small_t.cu` 的局部 `static_assert` 用的是 `kFp8FirstSmallT` ⇒ 自动放行✅；`bf16` 两处 `>=2` **未动**（bf16 不在本 diff 的路线里）✅。
   b. **workspace 一致性**：三处 planner 改动已在 §3 列出；`bf16_gdn_gating` 的 3840 B partial 由 `route_capacity` 自动纳入（已核对 `bf16_gdn_gating_proj_plan.cpp:276-288,370-380`）。
   c. **死代码**：`launch_a16`（`nvfp4_dispatch.cpp:50-66`、`nvfp4_linear_add_plan.cpp:30-46`、`fp8_linear_swiglu_plan.cpp:28-45`）在部分档位变为不可达；`Nvfp4LinearSwiGluRoute::DecodeFusedA16/SmallTFusedA16` 仍被 A16Only 分支使用；`fp8_gdn_conv_fused.cu:72-88` 的匿名 namespace 函数 `launch_snapshot_decode` 变为**无调用**（唯一一处可能产生 `-Wunused-function` 提示）。**本 diff 未删任何函数**（避免误删导致编译/链接问题）；这些 dead path 建议在下一轮单独清理。
   d. 仓库 `CMakeLists.txt` 未见 `-Werror`/`-Wall` ⇒ 匿名 namespace 内的未用函数不会致命；因此本 diff 选择"保留而非删除"。
2. **数值断言未实测**：§4 的"逐位一致"是**逐表达式**推导，不是实测。必须用 `_ga_check.sh` 的**首次偏离位置**做判决（预期：★ 项对应的首次偏离显著后移或消失）。
3. **同核但不同 split 数**只在 gating 项被消除；注意力项（§6.1）明确保留差异。
4. **W=2/4/16 的 verify 侧也变动**（§2 末）：若主代理只用 W=8 验收，请补一次 W=16 的接受率 + tok/s，确认 lm_head split-K 改动无副作用。

---

## 8. 需主代理实测的清单（按优先级）

1. **编译 + 冒烟**：`export PATH=/home/user/.local/bin:$PATH`（ccache，否则 127 且删二进制），全量重建，确认 17 文件编译通过、无新增 warning 影响。
2. **G-A（主判据）**：`_ga_check.sh`（zh + num）—— 看 **首次偏离位置**是否后移/消失（plain 侧数值变了是**预期**：T=1 迁移到 verify 的算术上；判据是两侧**相等**，不是"plain 不变"）。
3. **性能（硬门槛）**：
   - T=1 decode tok/s（改前/改后）；**门槛 ≤3%**，若超则按 §5 的粒度回退（先 #12 再 #5）。
   - **W=8 verify tok/s + 接受率**（预期 0 变化；若变说明我漏了 verify 侧某个 route）。
   - W=16 一次（覆盖 lm_head `kKWarps` 改动）。
4. **单变量定位（若 G-A 未达标）**：用 `--spec` 把 draft 宽度改成 4（verify 走 `SmallT`，与 T=1 同族）→ 若残差明显下降，说明剩余差异主要来自 §6.1 的注意力分档；若不变，注意力可降级处理。
5. **workspace 溢出检查**：用最小/最大 token 区间各跑一次 `linear_swiglu`/`linear_add`/`attn_input_proj` 的 warmup（阈值边界 T=1、4、5、7、8、11、16、48）确认无 "workspace is too small"。
6. （可选）若 §5 的 +0.27 ms 超预算：把激活量化融进上游 epilogue（不在本 diff 内）。

---

## 9. patch 应用与回退

```bash
export PATH=/home/user/.local/bin:$PATH
cd /home/user/ninfer-fusion
patch -p1 --dry-run < /mnt/c/Users/User/Documents/ziqinzhang/_collab/UNIFY_A_patch.diff   # 已验：exit 0，全部 hunk 精确命中（无 offset）
patch -p1          < /mnt/c/Users/User/Documents/ziqinzhang/_collab/UNIFY_A_patch.diff
```

- 回退：`patch -p1 -R < …/UNIFY_A_patch.diff`（diff 仅改 17 个源文件，无新增/删除文件）。
- **4 个文件是 CRLF**（`linear/fp8/fp8_config.h`、`linear/nvfp4/nvfp4_config.h`、`linear/nvfp4/nvfp4_dispatch.cpp`、`linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`）：hunk 中 CR 已按原样保留，故 `patch` 可直接应用（勿用会吃掉 CR 的工具二次处理该 diff）。
- diff 校验值：`md5 332a3662f47edfebb17065c5f90563a0`，373 行，17 文件。
