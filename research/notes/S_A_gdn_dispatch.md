# S_A：GDN kernel 分派（权重格式 × token 数）与 conv 补丁无效之谜

范围：只读分析 `/home/user/ninfer-fusion`（WSL 构建树）。未编译、未跑引擎、未改源码。
所有结论均给出 file:line。命令经 `wsl.exe bash -c` + 临时 .sh 执行（脚本已清理）。

---

## 0. TL;DR（先看这 5 条）

1. **本 artifact（NVFP4）的 GDN input projection 存在一条“精度悬崖”**：`width==1` → W4A16 GEMV；
   `width∈[2,3]` → W4A16 small-T GEMM；**`width>=4` → 走 Materialized：激活被量化成 FP4（W4A4）**
   + 另一个 conv kernel。即 verify 列数一旦 >=4，投影精度与 plain decode 就不是同一档。
2. **被补丁的 `GdnConvEpilogue`（gdn_conv.cuh）在 NVFP4 下只有 `width∈{2,3} && batch==1` 可达**；
   `width==1` 时补丁是死代码（它只喂“下一列”，而下一列不存在）；`width>=4` 时 conv 由
   `nvfp4_gdn_snapshot_post.cu` 的独立 kernel 完成，**那个文件根本不使用 `GdnConvEpilogue`**。
   ⇒ 这就是“落地 conv 补丁后 mtp 流逐位未变”的真正原因（且 K=3/K=7 → W=4/W=8 必然命中该情形）。
3. **排除“没重编”**：8 个 includer 的 .o 全部在 `.cuh` 修改后 13–33 秒被重编，且 `.o.d` 依赖文件
   确实记录了 `gdn_conv.cuh`。补丁**在**二进制里。handoff §1“无头文件依赖跟踪”对 `ninfer_ops` 不成立。
4. **GDN 递推在投机相关宽度上完全是逐 token**：chunked 只在 `T>=64` 才启用（kChunkSize=64）。
   W<=16 的 verify 永不进入 chunked 路径 ⇒ §4 的“chunked-vs-递推对齐”对本问题是**空任务**。
5. draft/MTP 模块**本身没有 GDN 层**（`dflash2_impl.h` 0 处 gdn）。若“mtp 的 token 流”指草稿自己的
   proposal 流，则它对任何 GDN 改动**天然不变**。
6. **（追加，见 §7）** `--prefill-chunk 128 vs 4096` 首 token 分叉：GDN 真正跨 chunk 携带的只有
   ①FP32 运行状态、②**BF16** `h_chunk` 快照、③**BF16** W/U/v_new、④**BF16** conv 窗口（但 conv 是
   逐位等价，属负发现）。**rope/positions 不是携带物**（每 chunk 现算绝对位置）。已定位到一处
   “已发布 BF16 vs 更宽累加器”的**形状相关**不一致：`gated_delta_net.cpp:254-261`。
7. **verify(T=W) 与 plain(T=1) 属于同一类**（token 数决定 kernel/route），且额外跨**精度档**（A16↔A4）
   与跨**算法**（fused↔materialized chunked conv）。最小改动排序见 §7.4。

---

## 1. 权重格式分派（QType → op 家族）

公开入口 `src/ops/wrapper/gdn_input_proj.cpp`。三个 API × 四种格式：

| API | 用途 | NVFP4 | FP8 | W8 | Q4Q5 |
|---|---|---|---|---|---|
| `gdn_input_proj` (wrapper:720/774) | 纯投影，**无 conv** | nvfp4_gdn_input_dispatch (wrapper:311) | fp8:331 | w8:347 | q4_q5 (wrapper:736) |
| `gdn_input_proj_conv_snapshot` (wrapper:910/1012) | conv + 就地更新状态 | nvfp4_gdn_snapshot_dispatch (wrapper:465) | fp8 (wrapper:511) | w8 (wrapper:549-567) | — |
| `gdn_input_proj_conv_record` (wrapper:965/1034) | conv + 只记录（回放用） | wrapper:610-632 | fp8:669 | w8:709-715 | — |

本 artifact = `qwen3_8_27b_nvfp4_dflash2.ninfer` ⇒ **只命中 NVFP4 列**。

**policy 已确证为 `AllowA4`**（不是 A16Only）：
`src/targets/qwen3_6_27b/impl/variant.cpp:64` `kNvfp4TextPolicy = LinearPolicy::AllowA4`；
`variant.cpp:67-70` `text_policy(NVFP4) → AllowA4`；调用点 `variant.cpp:316-318 / 341-343`。

---

## 2. token 数分派表（NVFP4 + AllowA4 的实际行为）

解析器：`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:28-45`

```
batch_size > 1            -> Materialized
A16Only: tokens == 1      -> DecodeFusedA16 ; tokens <= 16 -> SmallTFusedA16 ; 否则 throw
AllowA4: tokens == 1      -> DecodeFusedA16 ; tokens <= 3  -> SmallTFusedA16 ; 否则 Materialized
```

派发：同文件 `:68-90`；record 路径 `src/ops/wrapper/gdn_input_proj.cpp:610-632`。

| 场景 | width/batch | schedule | 投影 kernel | 激活精度 | conv kernel | 补丁是否在路径上 |
|---|---|---|---|---|---|---|
| plain decode（B=1） | W=1,B=1 | DecodeFusedA16 | `nvfp4_gemv_kernel`（nvfp4_gdn_snapshot_decode.cu:20） | **A16**(bf16) | `GdnConvEpilogue<1>` | **否**（Tokens=1，死代码） |
| verify W=2,3（B=1） | W=2..3 | SmallTFusedA16 | `nvfp4_small_t_kernel`（nvfp4_gdn_snapshot_small_t.cu:33） | **A16** | `GdnConvEpilogue<W>` | **是** |
| verify W>=4（B=1） | W>=4 | Materialized | `nvfp4_w4a4_mma_kernel`（nvfp4_gdn_input_w4a4.cu:28） | **A4**(激活量化成 FP4) | `nvfp4_gdn_conv_post_kernel`（nvfp4_gdn_snapshot_post.cu:15） | **否** |
| 任意 B>1 | batch>1 | Materialized | 同上（`AllowA4` 硬编码于 snapshot_plan.cpp:86；record 用调用方 policy wrapper:616/:629） | **A4** | `gdn_projected_conv_*`（gdn_projected_conv.cu:40-69） | **否** |
| prefill / 非 Verify 相位 | 任意 T | — | `gdn_input_proj` | 视 policy | `ops::causal_conv1d_silu_split`（text_context_impl.h:1135） | **否**（第 4 套实现） |

关键证据：
- `nvfp4_gdn_snapshot_plan.cpp:42-44` 是**唯一**把 AllowA4 限制在 `tokens<=3` 的地方；A16 分支给到 16。
- Materialized 分支 `:86` **硬编码** `LinearPolicy::AllowA4` → 必然走 W4A4。
- `nvfp4_gdn_input_w4a4.cu:40` `launch_nvfp4_w4a4_quantize(x, ...)` = 把**激活**量化成 NVFP4；
  同一文件 `:42-58` 再按 token 数分派 MMA M-tile（<=64→M32N64、<=96→M32N128、<=128→M128N128Pipelined、
  <=192→M64N128、否则 M128N128Resident；>=1024 且 256 整除走 TMA）。⇒ 归约树随 token 数变化。
- `nvfp4_gdn_input_plan.cpp:17-22` `resolve_route`：A16Only→A16、AllowA4→W4A4；
  `:29-46` `launch_a16` 按 `kNvfp4LastSmallT=32` 切块（nvfp4_config.h:197），块内 `active==1` 才用 decode。
- A16 路线的 fused GEMM 用的是 bf16 激活（`nvfp4_gdn_snapshot_small_t.cu:36` 传 `x.data` 为 bf16）。

**相位门**：`text_context_impl.h:1094` 只在 `Phase::Verify` 时走 conv-snapshot/record；
普通 decode 与 prefill 都走 `:1128-1136` 的 `causal_conv1d_silu_split`（第 4 套 conv 实现，与前 3 套无关）。
plain（非投机）decode = `ordinary_decode_batch`（`:777`，绑 `width=1`，`:816` 调 `run_layers(Phase::Verify)`）；
spec verify = `target_verify_batch_impl`（`:827`，绑 `width=W, columns=W*batch`，`:873`），
状态动作 `RecordForReplay` 由 `speculative_target_impl.h:15` 设定。
`text_context_impl.h:1103` 强制：`UpdateInPlace` 下 width 必须为 1（否则 throw）。

---

## 3. GDN 递推：逐 token 还是 chunked？

### 3.1 conv（4-tap depthwise）—— 三套实现，两套是分块
- **fused epilogue**（被补丁的那个）：`gdn_conv.cuh:85-119`，一次 `store<Tokens>` 内**严格逐列串行**，
  窗口 `s0/s1/s2` 全程留在寄存器；只有在 `token>=valid` 时 `continue`（`:87-97`）。
- **materialized post**：`nvfp4_gdn_snapshot_post.cu:74-108` —— **分块，TokenTile=8**（`:117`），
  `for (tile = warp; tile < token_tiles; tile += WarpsPerCta)`；块首用 `load_history`（`:77-84`）
  从 materialized BF16 缓冲**回读前 3 列**，块内再串行。这正是它需要“物化”缓冲的原因。
- **projected conv**（B>1 用）：`gdn_projected_conv.cu:40-69`，对整段 width **逐列串行**（无分块）。

### 3.2 delta-rule 状态递推 —— 逐 token（chunked 只在 T>=64）
`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp`
- `:212-225`：统一入口，`q.ne[2] != 1` → 转调 `:241` 的 chunked 入口；`T==1` → `launch_recurrent`。
- `:241-290`：`T_full = (T/kChunkSize)*kChunkSize`（`kChunkSize=64`，common.h:8）。
  `T_full>0` → `launch_chunked`；余下 tail → `launch_recurrent_inout`（`:286`）。
  **T<64 ⇒ T_full=0 ⇒ 只跑 recurrent tail ⇒ 等价于逐 token。**
- verify 走的两个入口都不是 chunked：
  - `UpdateInPlace` → `gated_delta_net_batch_update`，硬约束 `tokens==1 && B<=8`（`:108-127`，`kMaximumBatch=8`）；
  - `RecordForReplay` → `gated_delta_net_replay_record`（text_context_impl.h:1159）→ `launch_recurrent_record`
    （launch.h:41-45，纯逐 token）。
- **结论（负发现，可省主代理一次实验）**：W<=16 的 verify 与 B<=8 的 batch update 都不进 chunked；
  chunked 只服务 prefill（T>=64）。§4①“chunked-vs-递推对齐”在本问题上不存在待对齐项。

---

## 4. 为什么落地 conv 补丁后 mtp 流逐位未变

### 4.1 先排除“没重编”（重要，原结论可能建立在此之上）
实测（只读 ls/grep）：
- `gdn_conv.cuh` mtime `2026-09-11 14:31:09`；其 8 个 includer 的 .o 重编时间
  `14:31:25 / 14:31:27 / 14:31:28 / 14:31:32 / 14:31:33 / 14:31:33 / 14:31:33 / 14:31:42`
  （nvfp4_snapshot_decode、fp8_conv_fused、nvfp4_snapshot_post、q4_q5_conv_snapshot、
  nvfp4_snapshot_small_t、gdn_projected_conv、w8_decode、w8_gemm_splitk）。
- `build/.../nvfp4_gdn_snapshot_small_t.cu.o.d` 等 8 个 `.d` 文件**明确列出 `gdn_conv.cuh`**；
  全树有 561 个 `.d`。⇒ `ninfer_ops` 头文件依赖跟踪**是工作的**；
  handoff §1 的“无头文件依赖跟踪”说法与实测不符（至少对 ninfer_ops 不成立）。
- `build/apps/ninfer` mtime `14:35:36`，晚于全部重编。⇒ **补丁在二进制内**。

### 4.2 真正原因：分派（补丁的可达域极窄）
被改的语句在 `gdn_conv.cuh:118`（handoff 记作 `:116` 是补丁前的行号；插入两行注释后下移）。

补丁的语义：`s2 = bf16(p)` 相对 `s2 = p`，**只影响同一 `store<Tokens>` 调用内 token>=1 的列**
（token=0 用的 `s0/s1/s2` 全来自 `initial_base` 的 bf16 回读，`:76-78`）。因此：

1. **`Tokens == 1` ⇒ 补丁是死代码**。plain decode 走 `make_gdn_conv_output<1>`
   （`nvfp4_gdn_snapshot_decode.cu:23`），循环只跑一次，`s2=...` 之后再无人读。
2. **`width>=4`（AllowA4）⇒ 根本不在这个 kernel 里**。`nvfp4_gdn_conv_post_kernel`
   （`nvfp4_gdn_snapshot_post.cu:15-110`）自带一份 conv 循环，且：
   - `:96` `const float p = __bfloat162float(projected[projected_base + row]);` ← **p 从 BF16 缓冲读出**；
   - `:106` `s2 = p;`。
   即它**已经是 bf16 语义**，把补丁原样搬过去是**字面意义的 no-op**（p 已可被 bf16 精确表示）。
   `gdn_projected_conv.cu:53`（`p = __bfloat162float(projected[...])`）+ `:69`（`s2 = p`）同理。
   这两个文件虽然 `#include "gdn_conv.cuh"`，但只为拿 3 个 Publish 结构体
   （`gdn_conv.cuh:11-42`），**不使用 `GdnConvEpilogue`**。
3. ⇒ **唯一**能看到补丁的 nvfp4 场景是 `width∈{2,3} && batch==1`（SmallTFusedA16）。
   K 扫描里 **K=3→W=4、K=7→W=8 都落在 Materialized**，所以那两次跑必然逐位不变。
4. 附带：draft/MTP 模块**没有 GDN**。`dflash2_impl.h` 中 `gdn` 命中 0 次；
   `layouts_impl.h:690-728` 的草稿 proposal 容量只用 `dflash_attention` / `dflash_mlp` / 纯 linear，
   没有任何 conv_states / gdn op。若“mtp 的 token 流”指草稿自身流，它对任何 GDN 改动天然不变；
   只有**被 verify 的 target** 的 GDN 会通过 logits 影响接受链。

---

## 5. 最小对齐方案 + 可证伪预测

### 5.1 先修“精度悬崖”（1 行，收益最大）
`nvfp4_gdn_snapshot_plan.cpp:42-44`：把 AllowA4 分支的 `tokens <= 3` 放宽到与 A16 分支一致的
`tokens <= 16`（即 `if (tokens <= 16) return {SmallTFusedA16};`）。
- 该 schedule 已完全注册：`nvfp4_gdn_snapshot_small_t.cu:82-84` 的 launcher 表覆盖
  `kNvfp4FirstSmallT=2 .. 16`（nvfp4_config.h:196）；
- 调用方已就绪：`wrapper/gdn_input_proj.cpp:621-626` 已有 SmallTFusedA16 → `nvfp4_gdn_record_small_t_launch`
  分支；capacity 侧 `:450-451/:610-611` 自动跟随；
- 效果：verify 不再把激活量化成 FP4（`nvfp4_gdn_input_w4a4.cu:40`），且 conv 从“独立 post kernel”
  回到**与 plain 同一个** `GdnConvEpilogue`（FP32 当前列 + 已被补丁修好的 bf16 进位）。
- **这是“让 spec 与 plain 同精度档”的必要步骤**：A4 激活量化不可能通过任何下游 conv 修补变成逐位一致。

### 5.2 再做逐位一致（若要满足判据②“spec 流与 plain 逐位一致”）
即使 5.1 落地，`projected` 仍可能不一致，因为 T=1 与 T=W 用的是**不同 GEMM kernel / 归约树**：
- plain T=1：`nvfp4_gemv_kernel`（nvfp4_gdn_snapshot_decode.cu:20）
- verify W：`nvfp4_small_t_kernel`（nvfp4_gdn_snapshot_small_t.cu:33），累加链 `kAccumulatorChains`
  + `warp_reduce_sum`（nvfp4_small_t.cu:268-282），且 `projected` 是 **FP32 累加器**（`:279`，**未**经 bf16）。
⇒ 要达到逐位一致，必须让 T=1 列也由**同一个 kernel** 产出（表下界是 2，需注册 `ActiveTokens=1`），
或在验收标准上退一步：只要求“同一精度档 + 同一 conv 语义”，用 `_df2_blame2.py` 的逐列剖面
（p1+ 非零、AL>=3）而非逐位相等作为判据。

### 5.3 可证伪预测（建议按顺序做，都不需要改引擎逻辑）
1. **精度悬崖检验**：跑 K 扫描（`_ga_ksweep.sh`），若 5.1 之前 K=2（W=3，A16 fused）与
   K=3（W=4，Materialized/A4）之间存在**台阶式**（而非渐变）接受率/分歧位置跃变，
   则“悬崖”假设成立。handoff 已有 K=1→62、K=3→62、K=7→29 的分歧位置，但缺**接受率**随 K 的曲线。
2. **补丁可达性检验（最便宜、最决定性）**：在 K=1（W=2 ⇒ SmallTFusedA16，补丁必经）下比较
   补丁前/后的 spec token 流。**预测：必须逐位改变**。若实测仍逐位不变，
   则说明存在我未看到的第三条路径（或该次跑并非 W=2），需回头查 `drafts`/`selector_top_k`
   与实际 `active_sequence_width_` 的取值（`dflash2_impl.h`、`spec_decision.h`）。
3. **B 维度自洽检验**：B>1 走 Materialized(A4)，B=1 走 A16 fused ⇒ **引擎自身在 B 上不自洽**。
   用同一 prompt 跑 B=1 与 B=2（同一条序列占两槽）比较该序列的 logits；若不同，
   说明 A4/A16 分叉是真实的一阶偏差（而非纯舍入）。

---

## 6. 给主代理的落地清单（含精确坐标）
- 分派开关（唯一）：`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp:28-45`
  （`:36` batch>1、`:42-44` AllowA4 的 tokens<=3）。
- 4 套 conv 实现（改任一处都要同步另外 3 处才可能自洽）：
  1. `gdn_conv.cuh:84-119`（fused，已打补丁 `:118`）
  2. `nvfp4_gdn_snapshot_post.cu:74-108`（materialized，分块 8，`:96/:106`）
  3. `gdn_projected_conv.cu:40-69`（B>1，`:53/:69`）
  4. `causal_conv1d_silu`（prefill/非 Verify；`ops/wrapper/causal_conv1d_silu.cpp:277`）
- 精度分派：`nvfp4_gdn_input_plan.cpp:17-22`；`nvfp4_gdn_input_w4a4.cu:40`（激活量化）
  + `:42-58`（按 token 数的 M-tile 表）。
- 编译注意：`.cuh` 改后**确实**会触发重编（实测），无需手工 touch；但 `build/src/.../ninfer_ops`
  与 `build/apps/ninfer` 时间戳要一起核对（本次 14:31 重编 / 14:35 链接，链路正常）。

---

# 7. 追加：`--prefill-chunk 128 vs 4096` 首 token 分叉的跨 chunk 携带物清单

主代理运行期实证：同一 ~1200 token prompt，`--prefill-chunk 128` → 首 token 104980，
`--prefill-chunk 4096` → 首 token 103735（`_prefill_chunk_equiv.sh` / `dl/prefill_chunk_equiv.log`）。
`--prefill-chunk` 必须是 128 的倍数（`apps/cli/options.cpp:231-232`），默认 3072（`src/runtime/engine/engine.cpp:28`）。
每个 prefill 调用里 `T = x.ne[1]`（`text_context_impl.h:1080`）⇒ **chunk 大小直接决定每次 GDN/linear 调用的 T**。

## 7.1 跨 chunk 边界真正被携带的量（分类 + file:line + 精度）

| # | 携带物 | 载体 / dtype | 跨 chunk 语义 | file:line |
|---|---|---|---|---|
| ① | delta-rule 运行状态 `ssm_state` | **FP32**（宽） | 宽精度直接携带 | `gated_delta_net.cpp:80`（校验）、`:166`；`chunked/launch.h:78,82`；`chunked/state_passing.cuh:266` `load_ldg<float>(state_in…)` |
| ② | 每 chunk 边界状态快照 `h_chunk`，供 output 阶段消费 | **BF16**（窄，已发布） | 每 64 列发布一次 BF16 | `chunked/launch.h:42`（`DType::BF16 {128,128,H_v,chunks}`）；`chunked/launch.cu:55,68`；`chunked/output.cuh:11,155` |
| ③ | WY 中间量 `W` / `U` / `v_new` | **BF16**（窄） | 每 chunk 重算重发 | `chunked/launch.h:38-41`；`launch.cu:39-40,49-50,54`；`state_passing.cuh:54,90` |
| ④ | `g_cumsum` | FP32（宽） | 精确 | `chunked/launch.h:37`；`launch.cu:41,52` |
| ⑤ | **conv 3 列窗口** | `conv_state_in/out` **BF16** | **逐位等价**（不是嫌疑，见 7.3） | `ops/kernel/causal_conv1d.cuh:145-174`（`:169` 取值、`:171` 写出）；`:87-93`（回读） |
| ⑥ | snapshot 槽（**仅 Verify/record 路径**） | `conv_states` **BF16** ×3 列 | 已发布 bf16 | `gdn_conv.cuh:16-24`（`publish` 的 `:21-23` `__float2bfloat16_rn`）；`wrapper/gdn_input_proj.cpp:96-99`（dtype BF16, ne={C,3,cols,1}） |
| ⑦ | **rope / positions** | **无携带物** | 每 chunk 由绝对位置现算 | `text_context_impl.h:1359-1360` `fill_i32_positions(positions, base_i + t0)`；`:1381` `visible = base_i + t0 + len`；`:1412-1413` `io_.pos = base_i + T` / `io_.rope_pos = base_i + T + rope_delta_`；`MultimodalPrefill`（`text_context.h:382-388`）只传 `begin` 与整段 `positions`（`text_context_impl.h:1563`） |

⇒ **rope 不是本次分叉的嫌疑**：位置是 `base_i + t0` 的整数加法，与 chunk 切法无关（与 handoff 事实 7
排除 `rope_delta` 一致）。KV cache 也按绝对位置索引，无跨 chunk 数值携带。

## 7.2 已定位的“已发布（窄）vs 更宽累加器”不一致 —— 两处

### (a) q/k 归一化只有 T_full==0 时才留在 FP32（**形状相关，首要嫌疑**）
`src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp:254-261`：

```cpp
bool recurrent_normalize = normalize_qk;          // :254 默认 true
if (normalize_qk && T_full > 0) {                  // :255
    q_compute = scratch.normalized_q;              // :256 BF16 缓冲！
    k_compute = scratch.normalized_k;              // :257 BF16
    l2norm(q, 1.0e-6f, q_compute, stream);         // :258 归一化结果**写入 BF16**
    l2norm(k, 1.0e-6f, k_compute, stream);         // :259
    recurrent_normalize = false;                   // :260 告诉 kernel 别再归一化
}
```
- 缓冲区 dtype 明确是 **BF16**：`gated_delta_net.cpp:187-190`（`allocator.alloc(DType::BF16, …)`）。
- `T_full == 0`（本次调用不足 64 token）时不走 :255，`recurrent_normalize` 保持 **true** ⇒
  `launch_recurrent_inout(..., normalize_qk=true, …)`（`:286-288`）在 kernel 内用 **FP32 寄存器**归一化：
  `recurrent.cuh:64-70`（`rsqrtf(sum + eps)`，float）、调用点 `:125` / `:617`。
- ⇒ **同一个 token 的 q/k，归一化发生在「BF16 缓冲区」还是「FP32 寄存器」，只取决于把 prompt 怎么切成 chunk。**

代入本次实验（prompt ≈ 1200，不是 128 的整数倍）：
- `--prefill-chunk 128`：末尾一次调用 T = 1200 mod 128 = **48 < 64 ⇒ T_full = 0** ⇒ 末尾 48 个 prompt token
  走 **FP32 寄存器归一化**（`recurrent_normalize=true`）。
- `--prefill-chunk 4096`：单次调用 T = 1200 ⇒ `T_full = 1152 > 0` ⇒ 同一批末尾 48 token 由
  `q_tail = q_compute.slice(2, T_full, tail)`（`:276-277`）从 **BF16 缓冲**取值，且 `recurrent_normalize=false`。

末段 token 决定喂给首生成 token 的 GDN 状态 ⇒ 与“首 token 即分叉”的观测一致。这是本次唯一
**形状相关**的已发布/更宽不一致（其余携带物在两种 chunk 下的**绝对边界相同**：`T_full` 恒为 64 的倍数，
跨调用携带的 `ssm_state` 是 FP32，见 7.1①）。

### (b) `h_chunk`：chunked 路径的 chunk 边界状态以 BF16 发布给 output 阶段（**系统性偏窄**）
`chunked/launch.h:42` 分配 `h_chunk` 为 BF16，`launch.cu:55` 把 BF16 指针交给 state_passing，
`launch.cu:68` 再交给 output；`output.cuh:11` 的公式 `out = scale*(exp(g)*q@h_chunk^T + A@v_new)`
直接消费这个 BF16 状态。而 **recurrent 路径**用 FP32 状态在同一位置参与运算（`recurrent.cuh`）。
⇒ 与 `gdn_conv.cuh:118`（`s2 = bf16(p)`）同族：**一处消费已发布的窄值，另一处消费更宽的累加器**。
注：此项的边界恒在 64 的倍数，与 prefill-chunk 取值**无关**，因此不是本次分叉的直接原因；
但它是同一通式，建议与 (a) 一并修（否则 verify/recurrent 两条路永远不可能逐位一致）。

## 7.3 conv 是负发现：跨 chunk 携带 **bit-exact**，不是嫌疑
- `ops/kernel/causal_conv1d.cuh:74-101` `causal_conv1d_prefill_kernel`：每个 token **直接从 BF16 的 x 缓冲**
  读 `x[t-3] / x[t-2] / x[t-1]`（`:87-93`），只有 `t<3` 才回落到 `conv_state`（`:88,90,92`）；
  没有寄存器里的跨 token 窗口。
- `:145-174` `causal_conv1d_prefill_state_kernel`：把最后 3 列**原样拷贝**（`v = x[seq_pos-3]`，`:169`；
  `conv_state_out[s] = v`，`:171`），bf16→bf16 无精度损失。
- ⇒ 跨 chunk 的 conv 窗口 = 原始 BF16 列的拷贝 ⇒ **与 chunk 切法逐位无关**。
- 这同时解释了为什么 conv 只有在 **fused** 路径需要 `s2 = p` 修补：fused epilogue 把 **FP32 累加器留在寄存器**
  当窗口（`gdn_conv.cuh:99` 取 `p`、`:118` 进位），而 prefill conv 是**回读自己发布过的 BF16 列**
  （`causal_conv1d.cuh:87-93`）——后者天然自洽。**prefill conv 无需改动。**
- 与 T 有关的只是**选哪个 conv kernel**：`ops/wrapper/causal_conv1d_silu.cpp:269-273`
  （decode / smallt / prefill 三选一，按 T）——属 7.4-4 的“同档不同 tile”类。

## 7.4 与 verify(T=W) / plain(T=1) 的关系：**同一类，但更严重**

**同类**：两者都由 **token 数**选 kernel / route，不需要任何语义 bug 就能不等价。list of 无条件的形状依赖：
- 每个 NVFP4 W4A4 linear（QKV/o_proj/MLP/lm_head/GDN 投影）按 token 数选 route 与 M-tile：
  `ops/linear/nvfp4/nvfp4_dispatch.cpp:51`（`kChunk = kNvfp4LastSmallT` 分块）；
  `ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu:40`（`nvfp4_w4a4_tma_route(x.ne[1])`，**激活量化 kernel 本身随 T 换**）
  与 `:42-58`（`<=64→M32N64`、`<=96→M32N128`、`<=128→M128N128Pipelined`、`<=192→M64N128`、`else→M128N128Resident`、
  `>=1024 且 256 整除→TMA`）。T=128 与 T=1200 落在完全不同的分支。
- attention：`ops/launcher/gqa_attention_prefill.cu` vs `…_decode_smallt.cu` 等按 T 选 kernel。
- conv：`causal_conv1d_silu.cpp:269-273` 按 T 选 kernel。
- GDN：`gated_delta_net.cpp:212/241`（T==1 → recurrent；T>1 → chunked 入口，`T>=64` 才真 chunked）。

**更严重**：verify 侧还会跨**精度档**与跨**算法**——`width>=4` 时激活被量化成 FP4（§2 的悬崖：
`nvfp4_gdn_snapshot_plan.cpp:42-44` → `nvfp4_gdn_input_w4a4.cu:40`），且 conv 换成另一个 kernel
（`nvfp4_gdn_snapshot_post.cu`，§4.2）。所以 “T=W verify vs T=1 plain” 的不等价**包含**本次 prefill-chunk
现象的全部机制，外加精度档与算法两重跳变。

### 最小改动（按收益/风险排序，均给精确坐标）
1. **消掉 7.2(a)**（1 处，约 6 行，风险最低、最可能是本次首 token 分叉的直接原因）：
   删掉 `gated_delta_net.cpp:255-261` 的 BF16 预归一化分支，令 `recurrent_normalize` **恒等于** `normalize_qk`，
   使每个 token 的 q/k 归一化永远在 FP32 寄存器里完成。`validate_chunked`（`:160-170`）只校验形状，
   不要求输入已归一化 ⇒ 该改动对 chunked 路径安全（chunked 各 kernel 自带 `normalize` 语义，
   `recurrent.cuh:125/:617` 的 `Normalize` 模板参数即此开关）。
2. **消掉 7.2(b)**：`h_chunk` 由 BF16 → **FP32**，三处同步：
   `chunked/launch.h:42`、`chunked/launch.cu:55,68`、`chunked/output.cuh:155`（及 `:96-102` 的消费处）。
   这是让 chunked 与 recurrent 同精度档的必要条件。
3. **消掉 §2 的精度悬崖**：`nvfp4_gdn_snapshot_plan.cpp:42-44` 放宽到 `tokens <= 16`（理由与坐标见 §5.1）。
4. **残余（同档不同 tile）**：7.4 列的无条件按 T 选 kernel。这类**无法**用“改 dtype/改钳位”解决，
   只能统一 kernel 家族（例如让 verify 用与 plain 相同的 kernel 产 `projected`，见 §5.2）或接受非 bit-exact 验收。
   **建议验收顺序：先修 1–3，再复测；若残余分叉仍在，才去碰 4**，否则会把“精度档/发布口径”错误
   误判成“cmath 舍入不可控”。

### 可证伪预测（都不需要改引擎逻辑，按性价比排序）
1. **长度对齐实验（最便宜、最直接）**：把 prompt 补齐/截断成 **128 的整数倍**再跑同一对照。
   若 7.2(a) 是首因，则 (128 vs 4096) 的首 token **应变为一致**（此时不再存在 `T_full==0` 的调用）；
   若仍分叉，则主因是 7.4-4 的无条件 tile/route 切换，下一步查
   `nvfp4_gdn_input_w4a4.cu:40`（TMA 路线 vs M-tile 路线的激活量化布局是否不同）。
2. **单点修补验证**：只做 §7.4-1（6 行）后重跑同一对照；预测首 token 分叉消失或显著收敛。
3. **尾部长度扫描**：固定 `--prefill-chunk 4096`，把 prompt 长度从 128 的整数倍起每次减 1（或取
   `len mod 128 ∈ {0, 32, 48, 64, 96, 127}`），预测 **仅当 `len mod 128 > 0 且尾段 < 64` 时**
   (128 vs 4096) 才分叉；`len mod 128 == 0` 时一致。这把“形状不等价”精确锁定到 7.2(a)。
4. **对照 `_ga_check.sh`**：该脚本按 spec/plain 两条流比对，本次现象与投机无关；建议再补一条
   `--prefill-chunk` 维度（同 prompt、同 seed、只改 chunk）作为**回归判据**加入日常验收，因为
   它是最便宜的“形状不等价”探针（无需投机、无需改代码、单次跑即可暴露）。

