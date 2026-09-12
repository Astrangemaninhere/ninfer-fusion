# UNIFY-B：统一小 T 家族（T∈[1,16]）的算术 —— 逐分派点取证 + 统一 diff + 性能代价（2026-09-11）

树：`/home/user/ninfer-fusion`（只读分析 + 只产出 diff/report，**未编译、未跑 GPU、未改树**）。
产物：本报告 + `_collab/UNIFY_B_patch.diff`（7 文件 / 8 hunk，`patch -p1 --dry-run` **全部通过，exit 0**）。
范围：dflash2 verify（T=W∈{2,4,8,16}）与 plain 解码（T=1）在**每个 T 分派点**上的 kernel / 累加顺序 / 精度档。

---

## 0. 结论先行

1. **小 T 家族可统一，且不需要把任何一侧降级、也不需要额外 launch**：统一轴是
   **fp32 累加器的"分链方式"**，不是 kernel 家族。`nvfp4` 全部 A16 路径里，
   gemv（T=1）用 `AccumulatorChains=4`，small_t（T∈[2,16]）用 `AccumulatorChains=1`；
   两侧的 **lane→K 映射、scale 分组、warp 归约、epilogue 逐字相同**（§3 逐表达式对照），
   把 gemv 侧改成 1 条链后，**T=1 与 T∈[2,16] 逐位同算术**。这只需 **2 行**（两个 Schedule 的 `4 → 1`）。
2. 第二条轴是**路由档位**：`nvfp4` 各 op 在 T≥4/5/7/8 就切到 **W4A4（激活量化到 FP4）**，
   而 T=1 是 A16 ⇒ 家族被切成两档精度。本 diff 把 **T∈[1,16] 全部收进 A16 档**
   （verify 的 W=4/8/16 由 A4 升到 A16，精度**变高**而非降低），W4A4 只保留给 T≥17。
3. 性能方向与硬约束一致：**T=1 一行不改 kernel**（只减 3 条累加链 = 少 3 个寄存器级并行），
   **verify W=4/8/16 的 launch 数减少**（去掉 quantize 内核），权重访存量不变（每个权重字节仍只读一次）。
   估算：T=1 端 0~2%（ILP 被 16 warps/SM 的占用掩盖）；T∈[4,16] 端 −1 launch/op（更省）。
   **唯一有实际代价的是 GDN gating proj（E7）：T=1 从非协作 GEMV 变成协作 Split10 launch。**
4. **无法在不牺牲性能下统一的一处是 core attention**（T=1..6 走 `SmallT`、27B 的 W=8/16 走
   `Prompt`）——理由与替代见 §5。
5. **必须最先确认的一件事**：27B 的 `Qwen38Nvfp4DFlash2` profile 把 attn/gdn 输入投影映射到 **FP8**
   （`variant.cpp:435-439`、`:479-483`）。如果 live artifact 走的是该 profile，则本 diff 的
   nvfp4 编辑对 attn/gdn 两点**不生效**，需改 FP8 的同形表（§6 R7）。

---

## 1. 统一方案（一句话）

> 对每个按 tokens 分档的 nvfp4 分派点，**T∈[1,16] 只允许一种路由**；
> T=1 保留自己的 gemv 内核（最便宜的形状，不引入任何新 launch），
> 但把它的 fp32 累加**分链方式**改成与 T∈[2,16] 的 small_t 完全一致；
> 两侧的 kernel 家族差异（warp↔row 映射、staged vs direct scale 载入）已被证明**保值**。

谓词统一形态：`tokens <= 16 ? A16 : W4A4`（原为 `>= 4/5/7/8 ? W4A4 : A16`）。

---

## 2. 逐分派点表格（谓词 → 现状 → 统一）

| # | 分派点（file:line，谓词原文） | T=1 现状 | W=8/16 现状 | 统一后 | diff |
|---|---|---|---|---|---|
| E1 | `linear/nvfp4/nvfp4_config.h:193` `Nvfp4GemvSchedule<8,2,16,4,StagedRaw,Default,2>`（所有 decode：attn/gdn/linear/linear_add 共用） | gemv，**4 链** | small_t，**1 链** | 两侧都 1 链 ⇒ 逐位同 | ✓ |
| E2 | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu:16` `Nvfp4GemvSchedule<8,2,16,4,Direct,Default,2>` | 同上 | `nvfp4_linear_swiglu_small_t.cu:83` 1 链 | 两侧都 1 链 | ✓ |
| E3 | `linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp:46-47` `if (tokens<=48) FusedW4A4`（且 :40 `tokens<=4`→A16、:37 环境变量旁路） | `DecodeFusedA16`(A16) | `FusedW4A4`(A4, M48N64) | `T=1→DecodeFusedA16`；`T=2..16→SmallTFusedA16`(A16, 同一 small_t 系) | ✓ |
| E4 | `attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:24` `return tokens>=4 ? W4A4 : A16;` | `launch_a16`→decode gemv | W4A4（M32N64） | `T<=16 ? A16 : W4A4`（T=1 gemv / T=2..16 small_t） | ✓ |
| E5 | `linear/nvfp4/nvfp4_dispatch.cpp:31-39`（`>=4` / GdnInput 恒 W4A4 / `>=5` / `>=8`） | gemv | W4A4（M32N64） | `T<=16 ? A16 : W4A4`（GdnInput 一并收进 A16 档） | ✓ |
| E6 | `linear_add/nvfp4/nvfp4_linear_add_plan.cpp:26-27` `first_w4a4 = input_rows==6144 ? 7 : 8` | gemv | W4A4（M32N64） | `T<=16 ? A16 : W4A4` | ✓ |
| E7 | `gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:29-39` `k27Routes{ {1,1}→GemvPairedRows, {2,8}→SmallTSplit10, {9,1024}→MmaCooperativeSplit8 }` | GemvPairedRows | W=8→SmallTSplit10；W=16→MmaCoopSplit8 | `{1,16}→SmallTSplit10`（一个 kernel、一个 SplitK=10 归约分组）+ `candidate_is_legal` 放宽到 `cols>=1&&<=16`（:150） | ✓ |
| — | `gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:42-43`（`tokens==1→DecodeFusedA16; tokens<=16→SmallTFusedA16`，修②后仍如此） | gemv(A16, 融合 conv) | small_t(A16, 融合 conv) | **无需改**：E1 已让两者同算术（同一 `GdnConvOutput` epilogue） | 无 |
| — | `gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp:20` `if (policy==AllowA4) return W4A4;`（无 T 分档） | W4A4 | W4A4 | 已同档（T 无关） | 无 |
| Δ1 | `softmax_attention`：`wrapper/gqa_attention.cpp:404` `width 1..6 → SmallT`；`:408-412`（`q_heads==16` 才 `ChunkedSmallT`）；27B=24 heads ⇒ W∈[7,16] → `Prompt` | SmallT（split-K + fp32 acc/m/l） | Prompt（无 key 轴拆分） | **不统一**（§5） | 无 |
| Δ2 | `launcher/causal_conv1d.cu:264/274/284`（`B==1&&T==1`→decode；`T<=32`→smallt）；同样在 `wrapper/causal_conv1d_silu.cpp:268/270/273`、`:332/334/337`、snapshot `:264/274/284` | decode kernel | smallt/sequence kernel | **未做**（需先逐行核两个 kernel 的算术；见 §5） | 无 |
| Δ3 | `sparse_moe/prefill/sparse_moe_prefill_kernels.cu:1150-1153` `adaptive = tokens>=47 && tokens<=51/52` | — | — | 仅 35B-A3B（groupwise-int）命中；27B dense 不涉及（S_E §1.4） | 无 |
| — | `lm_head`、rope、rmsnorm/l2norm、KV fill、nvfp4 MMA 六档 schedule | — | — | **无分叉**（S_E §2.1/2.2/2.4/2.6/2.9 + R21 §二.2 已证负面结论） | 无 |

---

## 3. 量化证据：为什么"改 1 条链"就能逐位相同（逐表达式对照）

以 `Nvfp4GdnInputGeometry` / `AttnInput` / `Residual*` / `MlpGateUp`（K=5120 或 6144/17408）为例，
两侧 Schedule 差异只有 `AccumulatorChains`（4 vs 1）与 `ScaleAccess`（StagedRaw vs Direct）：

| 项 | gemv（T=1） | small_t（T∈[2,16]） | 相同性 |
|---|---|---|---|
| lane→K 值 | `nvfp4_gemv.cuh:182` `activation_index = phase*(kValuesPerPhase/2) + lane*kPairsPerLane + pair`，`kValuesPerPhase=32*16=512` | `nvfp4_small_t.cuh:166-168` `pair_index = phase*(kValuesPerPhase/2) + warp_in_row*(kValuesPerWarpPhase/2) + lane*kPairsPerLane + pair`，`kWarpsPerRow=1 ⇒ warp_in_row=0` | **逐字相同**（512=512，kPairsPerLane=8） |
| scale 分组 | `group = ((lane*kValuesPerLane & 15) + pair*2)/16 = 0`（lane*16&15≡0），系数 `decode_nvfp4_e4m3(scale)*inverse` | 同表达式（`small_t.cuh:157`） | **同值**（StagedRaw 只是把同样字节搬到 shared：`nvfp4_gemv.cuh:102-111` 的 `group_base+(lane&3)` 与 Direct 的 `group_begin=phase*32+lane` 同字节） |
| 累加 | `fmaf(code.x*coef, act.x, acc[...][(2*pair)&3])`：**4 条链** | `... [(2*pair) & (1-1)] = 0`：**1 条链** | ✗ ⇒ **本 diff 修此** |
| per-lane 串行深度 | 160 FMA / 4 链 = 40 | 160 / 1 = 160 | 改动后都是 160 |
| warp 归约 | `warp_reduce_sum`（`warp.cuh:21`，shfl_down 16,8,4,2,1） | 同一个模板、默认 Width=32 | **逐位相同** |
| 输出 | `sum_chain → warp_reduce_sum → __float2bfloat16_rn`（`nvfp4_output.cuh:23` / `nvfp4_gdn_input_output.cuh:28`） | `sum_chain → warp_reduce_sum → epilogue.apply`（`small_t.cuh:330-338`） | **逐位相同**（4 链改 1 后 `sum_chain` 变成 `((0+a)+0)+…` 恒等） |
| warp↔row 映射 | `m_tile*128 + rmod + quartile*32`（`nvfp4_gemv.cuh:222-229`，`rmod_base=cta_in_tile*(RowsPerCta/4)`） | 同公式（`small_t.cuh:248-254`，`rmod_base=cta_in_tile*(RowsPerCta/4)`） | **每个输出元素仍由唯一 warp 以同一 (lane,pair,phase) 序计算 ⇒ 逐位相同** |
| swiglu 收尾 | `silu(gate)*up` fp32→bf16（`decode.cu:52-61`） | `silu(gate)*up` fp32→bf16（`small_t.cu:57-71`） | **逐字相同** |
| swiglu MMA（T≥5 旧路） | — | 旧 `FusedW4A4` 的 `combine()` 是 `__bfloat1622float2` 读回 + `silu`（`w4a4.cu:41-50`） | A4 档，E3 后 T≤16 不再进入 |

**分派谓词取值（本用例，27B dense / nvfp4 / batch=1）**：
`attn_input: T=1→A16(gemv)；T=8,16→原 W4A4、改后 A16(small_t≤32 已注册，`nvfp4_attn_input_small_t.cu:96` 覆盖 2..32)`
`linear_add(6144/17408): T=1..16→改后 A16（原 T≥7/8 走 W4A4）`
`swiglu: T=1→DecodeFusedA16；T=2..16→改后 SmallTFusedA16（launcher 数组覆盖 2..16，`nvfp4_linear_swiglu_small_t.cu:104`）`
`gdn snapshot: T=1→DecodeFusedA16、T=2..16→SmallTFusedA16（修②已在位，脚本无改动）`

**可离线算出的数字（27B nvfp4，hidden=5120，nvfp4 = 0.5625 B/element）**
| 权重 | rows×K | 字节 |
|---|---|---|
| attn_input | 14336×5120 | 41.3 MB |
| gdn_input（融合） | 16384×5120 | 47.2 MB |
| MLP gate_up | 34816×5120 | 100.2 MB |
| o_proj | 5120×6144 | 17.7 MB |
| down | 5120×17408 | 50.2 MB |
| gating proj（bf16） | 96×5120×2 | 0.98 MB |
| **合计/层** | | **≈258 MB** ⇒ 60 层 ≈15.5 GB ≈ 8.6 ms @1.79 TB/s |

⇒ 这些 op 在 T≤16 上**全是权重带宽受限**（每 token 的算术量 ≪ 权重字节数），
所以"换 kernel 家族"本身≈0 代价，真正的代价只有 **launch 数**与 **ILP**。

---

## 4. 性能代价（每项改动）

| 改动 | 代价估算（roofline/访存量级） | 方向 |
|---|---|---|
| E1/E2 累加链 4→1 | 每元素 per-lane 串行 FMA 40→160；寄存器少 3×`RowsPerWarp` 个 fp32；占用不变（`__launch_bounds__` 8 warps × MinBlocksPerSm=2 ⇒ ≥16 warps/SM，FFMA 延迟被 warp 并行掩盖）。**估 0~2%**（仅作用于 T=1 的 gemv） | 略负 |
| E3 swiglu T=2..16 由 W4A4→A16 small_t | 权重字节不变；激活字节 +T*7680 B（T=16 时 +123 KB/op）；launch 数 2→1（去掉 `launch_nvfp4_w4a4_quantize`） | **正** |
| E4/E5/E6 T∈[4,16] 由 W4A4→A16 | 激活字节 +T*7680 B/op；launch 2→1。全链增量 ≈ 5 op×60 层×123 KB ≈ 37 MB vs 15.5 GB ⇒ **+0.24% 访存**，且少 5 次 launch/层 | **正/中性** |
| 精度档变化 | verify 激活由 FP4→BF16（**升档**）；logits 会整体变化（不再是"同数不同序"，而是"不同数"）⇒ 见 R2 | 质量↑ |
| E7 gating proj T=1 | GemvPairedRows（普通 launch）→ SmallTSplit10（`SplitK>1` 走 **cooperative launch**，`kernels.cu:279`）：+1 次协作 launch / GDN 层，估 +2~10 µs/层 ⇒ 32 层约 **+0.1~0.3 ms/step**。权重仍只 0.98 MB/层 | **需实测** |

**回退粒度**：E7 是唯一有可感代价的 hunk，且与其它 7 个 hunk 完全独立（不同文件/不同 op），
若 T=1 tok/s 退化 >1%，**只丢 E7** 即可（其余 6 点仍保持统一）。

---

## 5. 无法统一者及替代

### 5.1 core attention（T=1..6 `SmallT` vs 27B 的 W=8/16 `Prompt`）——**不统一**
- 谓词：`wrapper/gqa_attention.cpp:404-412`；27B 是 24Q/4KV ⇒ `q_heads==16` 的 `ChunkedSmallT`
  快路不适用（`:408`），W∈[7,16] 恒落 `Prompt`（`:412`）。
- 结构差异：`SmallT` 有 key 轴拆分的 partial（`small_t.cu` 的 fp32 `acc/m/l` + `gqa_attention_split_capacity`，
  `wrapper/gqa_attention.cpp:337-345` 分配 `FP32 m,l`），`Prompt` 无 key 轴拆分维
  （`softmax_attention/.../prompt.cu:49` grid = `div_up(tokens,Br) × QHeads`）⇒ 归约顺序必然不同。
- **为什么不能统一**：把 T=1 改成 `Prompt` = 单个 CTA 顺序扫完全部 visible keys（长上下文直接塌），
  违"禁止串行化"；把 W=8/16 改成 `SmallT` = 让 verify 走只对 ≤6 token 调优的分块/拆分配置，
  且 `SmallT` 的 split 数按 `count` 选（`:374-375`），W=16 会与 W=1 选到不同 split ⇒ **仍然不同算术**。
- **替代（同精度档）**：两条路已经同为"量化 KV + fp32 累积 + 精确 softmax"
  （`SmallT` 的 partial 是 FP32 `acc/m/l`；`Prompt` 为 fp32 累积器、无 key 轴拆分 ⇒ 每个 KV 位置的
  贡献只被加一次）。⇒ 建议把 attention 一列标注为"**已同精度档、非同归约序**"，
  G-A 判据里对 attention 段只要求"统计等价"。

### 5.2 causal_conv1d（T=1 `decode` vs T≤32 `smallt`）——**本轮未做**
- 谓词三处同构：`launcher/causal_conv1d.cu:264/274`（snapshot）与
  `wrapper/causal_conv1d_silu.cpp:268/270`、`:332/334`（silu/silu_split）。
- 统一方向（性能中性，符合"T=1 改走 T∈[2,32] 的 kernel"）：把 `x.ne[1]==1` 也路由到
  `causal_conv1d_smallt_launch`（`block=(ChannelTile, T=1)`，grid 不变），但**必须先逐行对照**
  `causal_conv1d_decode_kernel`（`causal_conv1d.cu:196`）与 `causal_conv1d_smallt_kernel`（`:96/:118`）
  的 4 抽头累加与状态发布顺序（修① 的教训：同一数学可能在发布精度上不同）。
  ⇒ 未进 diff，留给下一轮（工作量小，风险集中在"两 kernel 是否已等价"）。

### 5.3 FP8 / 其它权重档
同形的 T 分档表（`fp8_attn_input_plan.cpp:23`、`fp8_linear_swiglu_plan.cpp:25`、
`fp8_dispatch.cpp:42-58`、`fp8_gdn_conv_plan.cpp:45/88/106`）**未纳入本 diff**，
因为要先确定 live profile（见 R7）。

---

## 6. 残余风险

- **R1（最高）**：E1/E2 的"逐位相同"是**代码级论证 + 可证伪预测**，不是 GPU 实测。
  预测：应用 E1+E2 后，`_ga_check.sh` 的首次偏离位置**后移或消失**；
  且对同一 token 集，T=1 与 T=2..16 的同 op 输出应逐位相等。
  若实测不成立，说明我漏了一处差异，回退 E1/E2（1 行）。
- **R2**：E3-E6 把 verify 的激活档从 FP4 提到 BF16 ⇒ **verify 的 logits 会变**（不是"同数不同序"）。
  ⇒ 验收必须"整树 A/B"：新树重测 G-A、接受率、tok/s；不能拿旧树的翻转率直接比。
- **R3**：GDN 输入的 `Materialized` 路径用 `nvfp4_gdn_snapshot_post.cu` 手写 conv，
  与融合路的 `GdnConvOutput`→`GdnConvEpilogue`（`gdn_conv_output.cuh:22-46`）**是两段代码**。
  本 diff 统一的是融合族（`DecodeFusedA16`/`SmallTFusedA16`，共用 `GdnConvOutput`），
  T≤16 不进 `Materialized` ⇒ 不触及；但若将来 >16 的分派点也要统一，需先核这两段 conv。
- **R4**：E7 的 `SmallTSplit10 @ cols=1` 是**从未被路由过的组合**（原表 `cols>=2`）；
  且 `SplitK>1` 是协作 launch。需 GPU 校验：① 不崩、② T=1 tok/s。
- **R5**：`nvfp4_linear_swiglu_workspace_capacity_bytes` 对 `5<=max_tokens<=16` 仍按 W4A4 预算
  （`plan.cpp:103` 的 `max_tokens >= 5` 分支），而 T≤16 现在的路由不申请 workspace ⇒ **只多占内存、不减**
  （安全方向）。已刻意不改，避免动 planner。
- **R6**：树内 **CRLF/LF 混杂**（E1/E5/E2 所在文件是 CRLF）。patch 已按各文件行尾生成，
  `patch -p1 --dry-run` 对 7 个文件全绿；若主代理改用 `git apply`，需注意 CRLF 上下文。
- **R7（需先裁决）**：`src/targets/qwen3_6_27b/impl/variant.cpp:432-439` 把
  **`Qwen38Nvfp4DFlash2`** 的 attention 投影容量算成 **FP8**（gdn 同理 `:479-483`）。
  若 live 运行用的是该 profile，则 E4 对 attn_input、以及 gdn_input 的 nvfp4 分派点是**死代码**，
  真正要改的是 FP8 表。**请先确认 profile**（或直接从 schedule 日志里读实际命中的 dispatcher）。
- **R8**：R21/FIX_PLAN 的性能判据是"小 T 家族统一后 tok/s 退化 ≤3%"，本 diff 全部落在该预算内，
  但 **E1/E2 的 ILP 项与 E7 的协作 launch 项都需要实测**（见 §7）。

---

## 7. 需主代理实测的清单

1. **构建**：`export PATH=/home/user/.local/bin:$PATH`；`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`
   的 `k27Routes` 由 6 项改 5 项（`static_assert(catalog_is_closed)` 必须仍然过；已推演连续覆盖 1..∞）。
2. **先确认 live profile / 实际 dispatcher**（R7）：加一行 schedule 日志打印
   `attn_input/gdn/linear_add/swiglu/gating` 各自命中的 route（这正是 R21 §三 与 修② 一直缺的那行日志）。
3. **plain tok/s（T=1）**：zh + num，改前/改后。判据：退化 ≤1%（>1% ⇒ 丢 E7；仍>3% ⇒ 丢 E1/E2 并回报）。
4. **verify tok/s（W=8 与 W=16）**：同一 prompt/--spec dflash2。预期**不退化**（少 5 次 launch/层）。
5. **`_ga_check.sh`（spec vs plain，zh + num）**：记录首次偏离位置与全词表 argmax 翻转率。
   预期：E1/E2+E3-E6 后从 3.8% 明显下降（nvfp4 家族内的算术差被消掉）。
6. **接受率**：`_verify_df2head.sh`；注意 R2（verify 档位变化 ⇒ 基线会动）。
7. **单变量隔离（若 5 未达预期）**：只应用 E1+E2（2 行）先跑一次 G-A
   —— 这是"分链方式"这一假设的最小可证伪实验，且性能代价最小。
8. **E7 专项**：`--spec dflash2` 的 T=1 step 时延（vs 上一版）；若 gating proj 出现 launch 报错或非法配置，
   直接丢 E7 并在报告里记录（该点保持分叉）。
9. 若 profile 是 FP8（R7）：把 §5.3 的 FP8 表按同一谓词形态统一（`T<=16 ? A16 : A8`），再重跑 3~6。

---

## 8. 附：本 diff 的 8 个 hunk 一览（file:line 为改前行号）

| hunk | 文件:行 | 改前原文 → 改后 |
|---|---|---|
| 1 | `linear/nvfp4/nvfp4_config.h:193` | `Nvfp4GemvSchedule<8, 2, 16, 4, StagedRaw, Default, 2>` → `...16, 1, StagedRaw, Default, 2>` |
| 2 | `linear_swiglu/.../nvfp4_linear_swiglu_decode.cu:16` | `Nvfp4GemvSchedule<8, 2, 16, 4, Direct, Default, 2>` → `...16, 1, Direct, Default, 2>` |
| 3 | `linear_swiglu/.../nvfp4_linear_swiglu_plan.cpp:37-46` | 删掉 `NINFER_T1_W4A4` 旁路 + `tokens<=4→SmallTFusedA16`；新增 `tokens==1→DecodeFusedA16`、`tokens<=16→SmallTFusedA16` |
| 4 | `attn_input_proj/.../nvfp4_attn_input_plan.cpp:24` | `tokens >= 4 ? W4A4 : A16` → `tokens <= 16 ? A16 : W4A4` |
| 5 | `linear/nvfp4/nvfp4_dispatch.cpp:31-39` | 5 个 case 合并 → `tokens <= 16 ? A16 : W4A4` |
| 6 | `linear_add/.../nvfp4_linear_add_plan.cpp:26-27` | `first_w4a4 = 6144?7:8` + `>=` → `tokens <= 16 ? A16 : W4A4` |
| 7 | `gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:29-39` | `array<RouteSpec,6>` + `{1,1}Gemv/{2,8}SmallT10/{9,1024}Split8` → `array<RouteSpec,5>` + `{1,16}SmallTSplit10/{17,1024}Split8` |
| 8 | 同上 `:150` | `SmallTSplit10: cols >= 2 && cols <= 8` → `cols >= 1 && cols <= 16` |

`patch -p1 --dry-run`（cwd=`/home/user/ninfer-fusion`）输出：
```
checking file src/ops/linear/nvfp4/nvfp4_config.h
checking file src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
checking file src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
checking file src/ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
checking file src/ops/linear/nvfp4/nvfp4_dispatch.cpp
checking file src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
checking file src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
exit 0
```
