# FIX_A 报告：修① / 修②（+可选修③）— A 路（与 B 路独立同题）

作者：A 路子代理。只读源码 + 离线 CPU 计算；**未编译、未跑引擎、未用 GPU、未改动构建树**（三份目标文件 md5 在
dry-run 前后一致，见 §5）。命令经 `wsl.exe bash -c` + 临时 .sh 执行（脚本已清理；可复现产物见 §5.3）。

---

## 0. 结论先行（10 条）

1. **修① 定位比 S_B 更强**：`gdn_conv.cuh:99` 的 FP32 `p` 不只是"记录/状态 vs 输出不一致"，它是 **plain(T=1) 与
   spec(W≥4) 之间 conv 输出的直接不对称**。plain decode 走 `GdnConvEpilogue<1>`（用 24 位累加器算 conv 输出），
   spec verify 的 W≥4 走 materialized post kernel（`p` 从 BF16 缓冲读、天然 9 位）。**修①后两侧的 conv 算术逐项同序同值。**
2. **修② 的量级远大于修①**：W∈[4,16] 的 verify 目前把 **GDN 投影的激活量化成 NVFP4**（`nvfp4_gdn_input_w4a4.cu:40`），
   离线实测对 K=5120 的激活向量相对 L2 误差 = **9.48%**，单次点积相对误差 1.1e-1（§3.2）。改为 A16(bf16) 后，
   同口径误差 = **0.17%**。这是 1e-1 级的第一序不一致，**不是舍入**。
3. **修② 的"家族内同算术"可以证明（结构级）**：A16 small-T kernel 的 per-(row,token) FP32 累加器与 **ActiveTokens 无关**
   （K 维归约序由 (phase, lane, pair) 唯一决定，`kWarpsPerRow=1`/`kValuesPerLane=16`/`kValuesPerPhase=512`/10 phases；
   T==2 的 `SharedPhase` 与 T>2 的 `TokenPacked` 读的是**同一批字节**，`nvfp4_small_t.cuh:129-133`）。⇒ T∈[2,16] 内
   同一列的 `projected` 逐位相同。
4. 修② 后 **没有任何 T∈[1,16] 会进入 `nvfp4_gdn_input_w4a4.cu`**，所以计划书 §2 修② 要求的"w4a4 激活量化 kernel 与
   M-tile/TMA 表在小 T 家族内一致"**自动成立，该文件无需改动**（理由与反证见 §3.3）。
5. 修① 的改动是 **一行**（`const float p = projected[token];` → `const float p = __bfloat162float(__float2bfloat16_rn(projected[token]));`），
   对 `publish` 的落盘位**完全中立**（`SnapshotHistoryPublish`/`RecordColumnPublish` 本来就写 `bf16(p)`，`gdn_conv.cuh:21-23,35`），
   只把 conv 的第 4 抽头从 24 位换成 9 位 ⇒ "输出 = 状态 = 记录"三者同源。
6. 修② 是 **一行**（`nvfp4_gdn_snapshot_plan.cpp:43` 的 `tokens <= 3` → `tokens <= 16`），阈值放宽到 A16 分支已有的
   `tokens <= 16`（`:39`）；launcher 表 `nvfp4_gdn_snapshot_small_t.cu:82-84` 已注册 `kNvfp4FirstSmallT=2..16`
   （`src/ops/linear/nvfp4/nvfp4_config.h:196`），**record 侧** `wrapper/gdn_input_proj.cpp:621-626` 分支已存在 ⇒ 零新增代码。
7. **性能**：修① = +2 条 cvt/元素（conv epilogue 是访存受限），**中性**。修② = 权重流量不变的换路（fp4 MMA → FFMA
   小 T kernel，权重 26 MB/层/轮 相同），激活流量 ×4（5120×T×2B，可忽略）；**残余风险是从 tensor core 换到 CUDA core
   的算力比，必须实测 W=16 的 tok/s**（§4，这是本报告最需要主代理实测的一条）。
8. 修③ 给出两份 diff（可选）：`gated_delta_net.cpp:185`/`:255` 两处。**性能非严格中性**（T∈[2,63] 的调用 +2 次 l2norm
   启动 + 2×2048×T×2B 流量；解码稳态 T=1 不经过该函数入口，**解码 tok/s 不受影响**）。
9. **修③ 的判据（复核后）很可能可达——我撤回"不可达"的初判**：`T_full=(T/64)*64`。chunk=128 时前 9 次调用
   T=128 ⇒ T_full=**128**、tail=0（**全部进 chunked**），最后 1 次 T=48 ⇒ T_full=0；chunk=4096 时 T=1200 ⇒ T_full=1152、
   tail=48 ⇒ **两者的 chunked/recurrent 划分完全相同**（18 个 64 块 + 尾 48 token）。且
   ① `g_cumsum` 是**每 64 块内重置**的 Hillis-Steele 扫描（`chunked/prepare_wy_wu.cuh:393-442`：`cs=chunk*BT`、
   `ex_prefix=(lane==0)?0.0f`）⇒ 与"调用切分"无关；② `l2norm` 是**逐 (head, token) 行**归一
   （`include/ninfer/ops/l2norm.h:9-21`）⇒ 与 T 无关。修③前唯一的差别就是**末次调用（T=48, T_full=0）走 FP32 寄存器
   归一化**，而 4096 口径下同一批 token 走 BF16 已发布值（`:255-261`）。⇒ 修③ 有机会拿到 **IDENTICAL**；
   残余不确定项只剩"chunked 各块是否只通过 FP32 状态耦合"（未逐行核对，见 §6.2-7）。判据宜写"IDENTICAL 或显著后移"。
10. 残余**: T=1 用 `nvfp4_gemv_kernel`、T∈[2,16] 用 `nvfp4_small_t_kernel`——两个**不同 kernel**。修①+修②后 conv 与状态
    语义已完全对齐，但 `projected` 的 K 维归约树仍不同（差 ~1e-7，bf16 舍入后偶发 ±1 ulp）⇒ **spec 不可能逐位等于 plain**，
    除非再统一 T=1 的 GEMM（S_A §5.2 的结论成立）。这是本修复的硬上界。

---

## 1. 修①：fused conv 第 4 抽头改用已发布口径

### 1.1 改前/改后（原文）

`/home/user/ninfer-fusion/src/ops/gdn_input_proj/gdn_conv.cuh`（LF 行尾，ASCII）

```cpp
// 改前 :99
            const float p              = projected[token];
// 改后 :99
            const float p              = __bfloat162float(__float2bfloat16_rn(projected[token]));
```

未改动的三个消费者（`file:line` + 原文）：

| file:line | 原文 | 说明 |
|---|---|---|
| `gdn_conv.cuh:104` | `conv                       = fmaf(w3, p, conv);` | 唯一被本次改动影响的算术：抽头从 24 位变 9 位 |
| `gdn_conv.cuh:114` | `publish.publish(token, batch_row, row, s1, s2, p);` | 落盘位不变（见 1.2） |
| `gdn_conv.cuh:118` | `s2 = __bfloat162float(__float2bfloat16_rn(p));` | 已落地补丁；改后为幂等（`bf16(bf16(x))==bf16(x)`），**保留不改** |

### 1.2 为什么与"记录/状态"保持同源（同一行就能证明）

- `gdn_conv.cuh:21-23`（`SnapshotHistoryPublish`）：`__float2bfloat16_rn(s1)` / `(s2)` / `(p)`。
- `gdn_conv.cuh:35`（`RecordColumnPublish`）：`record[column * channels + row] = __float2bfloat16_rn(p);`
- ⇒ publish 侧本来就只发布 `bf16(p)`。**改前**：`conv` 消费 `p` 的 24 位、状态/记录存 `p` 的 9 位 ⇒ "验证列的输出
  对应一个状态里不存在的数"。**改后**：三者是同一个 `float`（bf16 可精确表示）⇒ 自洽。落盘位一个 bit 都没变。

### 1.3 触发谓词与取值（哪些路会走到）

`GdnConvEpilogue<Publish>::store<Tokens>` 的实例化点（`grep -n GdnConvEpilogue` 全量 25 处，关键如下）：

| 实例化点 | Tokens | 走的相位 / 动作 | 修①是否改变数值 |
|---|---|---|---|
| `nvfp4/nvfp4_gdn_snapshot_decode.cu:23`（`make_gdn_conv_output<1>`） | 1 | **plain decode**：`Phase::Verify` + `UpdateInPlace` ⇒ snapshot ⇒ `DecodeFusedA16` | **是**（T=1 时 `:118` 是死码，但 `:104` 的抽头活着） |
| `nvfp4/nvfp4_gdn_snapshot_small_t.cu:33-40`（`make_gdn_conv_output<ActiveTokens>`） | 2..16 | **spec verify**：`Phase::Verify` + `RecordForReplay` ⇒ record ⇒ `SmallTFusedA16`（W≤3；**修②后 W≤16**） | **是** |
| `q4_q5/q4_q5_gdn_input_conv_snapshot.cu:66,78,93,110` | 按 W | Q4Q5 权重格式的 conv snapshot/record | 是（另一权重格式） |
| `w8/w8_gdn_input_decode.cu:13,28`、`w8/w8_gdn_input_gemm_splitk.cu:33` | 1 / W | W8 权重格式 | 是（另一权重格式） |
| `gdn_conv_output.cuh:24`（结构体字段）、`:36`（z 通道） | — | z 行不经过 conv：`z[...] = __float2bfloat16_rn(projected[token])` | **否**（z 本来就 round 到 bf16） |

**不会**走到这里（即 S_A §4.2 的负发现，已复核）：`nvfp4/nvfp4_gdn_snapshot_post.cu:96`
`const float p = __bfloat162float(projected[projected_base + row]);`（p 从 **BF16 缓冲**读出）+ `:106 s2 = p;`；
`gdn_input_proj/gdn_projected_conv.cu:53` + `:69` 同形。这两个文件只 `#include "gdn_conv.cuh"` 取 3 个 Publish 结构体。

**落在哪条相位（判定它的确切代码行）**：`text_context_impl.h:1094`
`if (ph == Phase::Verify) {` → `:1118-1126` `gdn_input_projection_record`（`gdn_state_action_ == RecordForReplay`，`:1113` 判据）
vs `:1128-1135` `gdn_input_projection_snapshot`；非 Verify（= Prefill）走 `:1135`
`ops::causal_conv1d_silu_split`（第 4 套 conv，**与修①无关**，见 S_A §7.3）。

### 1.4 数值差量级（改前 vs 改后）

被消除的舍入：**一次 bf16 RNE 施加在 `p` 上**。离线实测（`_collab/_fixa_quant.py`，纯 CPU）：
- 逐元素相对误差上界 `2^-9 = 1.953e-3`；在"尾数在 ulp 内均匀"的模型下 RMS = **1.593e-3**（实测 200k 样本）。
- 该扰动经 `w3` 进入 conv 预激活：`|Δconv|/|conv|` 中位数 **3.56e-4**、p90 **4.10e-3**（20k 随机样本）。
- conv 输出经 `bf16(silu(conv))`（`:105`）发布，bf16 输出的相对量化步长 = 2^-9..2^-8 = **1.95e-3..3.91e-3**
  ⇒ 与 p90 的 Δconv **同量级** ⇒ 受影响的 (通道,列) 有 **O(10%)** 会翻转 1 个 bf16 ulp。这与"列 0 只有 ~85% 一致"
  的观测**相容**（不是巧合：`p` 是"本列输出 + 状态窗口"的共同来源，误差恰好落在输出量化步长附近）。
- 传导：窗口 `s2` 存的是 `bf16(p)`，**修①不改变窗口**，所以影响是**逐列局部**的（不累积到递推）。

**另需注意（不是修①能修的）**：J 侧：plain(T=1) 与 spec(W≥4) 用的 conv **算术序完全相同但输入 `p` 的来源不同**
（gemv vs small_t/mma 的 K 归约树）⇒ 残余 ~1e-7 相对，bf16 舍入后偶发 ±1 ulp。见 §6 残余。

### 1.5 性能

单条 `p` 定义处新增 `cvt.rn.bf16.f32` + `cvt.f32.bf16`（2 条 SASS，逐 (row, token)）。该 epilogue 是访存受限
（每 token 每通道一次 BF16 读 `state_read` + 一次 BF16 写 q/k/v + 每 token/通道一次 publish 写）。**无结构变化、无额外内存、
无额外 kernel、无额外 launch ⇒ 性能中性**。`:118` 的 `__float2bfloat16_rn(p)` 保留（幂等，编译器可折叠）。

---

## 2. 修②：统一小 T 家族（T∈[1,16]）的 GDN 路线与精度档

### 2.1 改前/改后（原文）

`/home/user/ninfer-fusion/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp`

```cpp
// 改前 :42-44
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= 3) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    return {Nvfp4GdnConvScheduleId::Materialized};
// 改后 :42-44
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    return {Nvfp4GdnConvScheduleId::Materialized};
```

### 2.2 触发谓词与取值

`nvfp4_gdn_conv_resolve_plan(policy, tokens, batch_size)`（`:28`）的判定顺序：

| 行 | 原文 | 取值 |
|---|---|---|
| `:30-32` | `batch_size > 8` → throw | — |
| `:36` | `if (batch_size > 1) { return {Materialized}; }` | **batch>1 恒 Materialized**（不受修②影响） |
| `:37-40` | `policy == A16Only` → `tokens==1`→Decode, `tokens<=16`→SmallT, else throw | A16Only 分支**本来就是 16** ⇒ 修②对它零影响 |
| `:42-44` | `AllowA4`（改前 `<=3`） | **受影响面 = `policy==AllowA4 && batch==1 && tokens∈[4,16]`** |

`policy` 取值来源（S_A 已确证，本报告复核为源文件常量）：`src/targets/qwen3_6/impl/variant.cpp:64`
`kNvfp4TextPolicy = LinearPolicy::AllowA4`。**所以该 artfact(qwen3_8_27b_nvfp4_dflash2) 必然命中放宽后的分支。**

两个消费者（都走同一个 plan 函数）：
- snapshot：`nvfp4_gdn_snapshot_plan.cpp:68-78` → `:74-78` `SmallTFusedA16` ⇒ `nvfp4_gdn_snapshot_small_t_launch`
  （替换掉 `:83-89` 的 `allocate_workspace` + `nvfp4_gdn_input_dispatch(..., AllowA4, ...)` + `nvfp4_gdn_snapshot_post_launch`）。
- record：`src/ops/wrapper/gdn_input_proj.cpp:610-632` → `:621-626`
  `if (plan.schedule == SmallTFusedA16) { nvfp4_gdn_record_small_t_launch(...); return; }`
  （替换掉 `:628-631` 的 `gdn_input_proj(x, weight, conv_record, z, policy, ...)` + `nvfp4_gdn_record_post_launch`）。

### 2.3 "小 T 家族内同路线、同精度档"的三条证据

1. **精度档**：改后 T∈[1,16] 的激活**从不被量化**——T=1 走 `nvfp4_gemv_kernel`（bf16 激活，`nvfp4_gdn_snapshot_decode.cu`），
   T∈[2,16] 走 `nvfp4_small_t_kernel`（`nvfp4_snapshot_small_t.cu:36` 传 `x.data`，bf16）。而 `nvfp4_gdn_input_w4a4.cu:40`
   `launch_nvfp4_w4a4_quantize(x, weight, workspace, nvfp4_w4a4_tma_route(x.ne[1]), stream)` 只在 `Materialized` 分支（`:79-89`）
   被调用 ⇒ **改后 T∈[1,16] 无一进入 w4a4**。
2. **家族内同算术**：`nvfp4_small_t_kernel` 的每 (row, token) 累加器与 `ActiveTokens` 无关：
   - `nvfp4_config.h:220-236`（`Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens>`）：
     `kWarpsPerCta=4`（与 T 无关）、`kValuesPerLane=16`（T<=16 恒 16）、`kTokenTile==ActiveTokens`（仅"列数"）、
     `kActivationAccess` 在 **T==2** 为 `SharedPhase`、T>2 为 `TokenPacked`。
   - `nvfp4_small_t.cuh:62-67`：`kValuesPerPhase = kWarpsPerRow(1) * (32 * kValuesPerLane(16)) = 512`，
     `kPhases = 5120/512 = 10`；`:107-199` 的 K 维累加序由 `(phase, lane, pair)` 唯一决定，**与 ActiveTokens 无关**。
   - 两种 access 策略读**同一批字节**：`TokenPacked` `pair_index = phase*(kValuesPerPhase/2) + warp_in_row*(..) +
     lane*kPairsPerLane + pair`（`:166-168`）；`SharedPhase` `local_pair = warp_in_row*(..) + lane*kPairsPerLane + pair`
     （`:178-179`），而其 staging 源地址 = `x + token*kInputRows + phase*kValuesPerPhase + local_pack*8`（`:82-85`）；
     两处都经 `bf16x2_bits_to_float2` 转换（`:176-185`）⇒ 同样的 bf16 pair。
   - 终局：`:267-284` `total = Σchains` → `warp_reduce_sum(total)` → `projected[local_token] = epilogue.apply(parent_row, local_token, total)`，
     **`projected` 是 FP32 累加器**（未 bf16 化）——这正是修① 存在的原因，也是 S_A §5.2 所指。
   ⇒ **T∈[2,16] 内，同一 (row, column) 的 `projected` 逐位相同。**（结构推断，未实测：见 §6.1）
3. **launcher 覆盖与容量**：`nvfp4_gdn_snapshot_small_t.cu:82-84`
   `make_index_sequence<16 - kNvfp4FirstSmallT + 1>`，`kNvfp4FirstSmallT=2`（`nvfp4_config.h:196`）⇒ snapshot/record
   两套 launcher 都注册 `ActiveTokens=2..16`；`static_assert(ActiveTokens >= 2)`（`nvfp4_small_t.cuh:215`）不触发。
   工作区容量自动跟随：`nvfp4_gdn_snapshot_plan.cpp:55` 与 `wrapper/gdn_input_proj.cpp:907-908`
   都是"`maximum_plan` 不是 Materialized ⇒ 返回 0" ⇒ 改后 W≤16 的 record/snapshot 容量**降为 0**（不再分配 w4a4 中间缓冲），
   是**缩小**而非放大，不改变任何 profile 的 arena 上界。**无需改任何容量函数。**

### 2.4 数值差量级（改前 vs 改后）

被消除的**不是舍入**，而是 NVFP4 **激活量化**（E2M1 codebook + per-16 e4m3 block scale，`nvfp4_codec.cuh:14-18`
用 `__nv_fp4x2_e2m1`，block scale 见 `Nvfp4QuantizedK16` `:26-30`）。离线实测（`_fixa_quant.py`，N(0,1) 激活，K=5120）：

| 口径 | 激活向量相对 L2 误差 | 单次点积相对误差 | 25 次 256 宽点积 |
|---|---|---|---|
| FP4(E2M1+per-16 scale) | **9.48%** | **1.14e-1** | median **5.1%**, p90 **16.6%** |
| bf16 RNE（=A16） | **0.17%** | **1.28e-3** | — |
| 单元素最坏（E2M1 相邻值比 3/2 ⇒ 1/6） | 16.7% | — | — |

⇒ 修② 消掉的是 **1e-1 级的投影不一致**，比修① 的 2e-3 高**两个数量级**。这与观测到的
"K=3(W=4)/K=7(W=8) 命中 Materialized 后分歧位置大幅提前（62 → 29）"方向一致。
（注意：两者都是"相对该量自身"的估计；实际 GDN 是 l2norm 后的 q/k 进递推，绝对影响被 ||q||·scale 缩放。）

### 2.5 性能

| 项 | 改前（W∈[4,16]） | 改后 | 变化 |
|---|---|---|---|
| 权重流量 | `nvfp4_w4a4_mma_kernel` 读全部 fp4 权重（10240×5120/2 B ≈ 26 MB/层/轮） | `nvfp4_small_t_kernel` 读**同一批** fp4 权重 | **不变**（两侧都是行全覆盖、每行读一次） |
| 激活流量 | `launch_nvfp4_w4a4_quantize` 写回 + MMA 读 fp4 codes（5120×T×0.5 B）+ 量化 kernel 启动 | 直接读 bf16 激活（5120×T×2 B） | 5120×16×1.5 B ≈ **123 KB/层**，可忽略；**省掉 1 个 kernel 启动** |
| 中间缓冲 | `allocate_workspace`（`projected` 10240×T bf16）+ w4a4 workspace | 无 | **省掉**（容量降为 0） |
| 算力 | tensor core fp4 MMA | CUDA core FFMA（每 code pair 2 次 `fmaf`） | **风险项**：FLOPs 相同，但吞吐率量级不同 |

⇒ **本项性能不是"显然中性"**，必须实测（§6.1）。缓解论证：(a) 权重流量（最可能的瓶颈，26 MB/层）不变；
(b) `nvfp4_config.h:199-201` 的注释明说这些 schedule 是 "RTX 5090 cold-cache winners for contiguous Linear output"，
即**在本卡上被调优过的生产 schedule**（且 A16 策略本来就把 T≤16 交付给这条路）；(c) 需要量的只有 W=8/16 的
encode/verify tok/s。**若退化，回退档位（保留一行改动不变、只把阈值收到 8 或 4）**：`tokens <= 8`。

---

## 3. 修③（可选：给出，但**建议单独提交**做单变量复测）

### 3.1 改前/改后（原文，两处同文件）

`/home/user/ninfer-fusion/src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp`

```cpp
// === 改前 :182-193（allocate_chunked_workspace 内） ===
    ChunkedWorkspace out;
    const std::int32_t full =
        (tokens / detail::gated_delta_net::kChunkSize) * detail::gated_delta_net::kChunkSize;
    if (full == 0) { return out; }          // ← 改前：T<64 时连 normalized_q/k 都不分配
    if (normalize_qk) {
        out.normalized_q =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
        out.normalized_k =
            allocator.alloc(DType::BF16, {detail::gated_delta_net::kStateDim, qk_heads, tokens});
    }
    out.stage =
// === 改后：把 `if (full == 0) { return out; }` 移到 normalize 分配之后 ===
    ChunkedWorkspace out;
    const std::int32_t full = ...;
    if (normalize_qk) { ...同上... }
    if (full == 0) { return out; }
    out.stage =

// === 改前 :255 ===
    if (normalize_qk && T_full > 0) {
// === 改后 :255 ===
    if (normalize_qk) {
```

### 3.2 触发谓词与取值（判定它的确切代码行）

- 该分支只在**11 参 in/out 重载**里（`gated_delta_net.cpp:241`）；唯一调用点
  `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1174-1175`
  `ops::gated_delta_net(q_recurrent, k_recurrent, vv, g, beta, kGdnScale, /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);`
  —— 它位于 `if (ph == Phase::Verify) {...} else {...}` 的 **else**（`:1169` 的 `else`）⇒ **只有 `ph != Verify`（即 Prefill）会到达**。
- `Phase::Verify` 的两条路都与修③无关：`text_context_impl.h:1157-1162`
  `gated_delta_net_replay_record`（RecordForReplay）与 `:1163-1166` `gated_delta_net_batch_update(..., /*normalize_qk=*/true, ...)`
  （UpdateInPlace，`:1103` 强制 width==1）⇒ **plain decode（`ordinary`，`layouts_impl.h:510` 绑 `Phase::Verify`+`Snapshot`）
  与 spec verify 都不受影响**。
- T=1 也不到该分支：`:216` `if (q.ne[2] != 1) { ... }` 把 T==1 路由到 `launch_recurrent`。
- ⇒ **受影响集合 = Prefill 调用且 T∈[2,63]**（只有 `T_full==0` 的那次尾调用行为改变；`T_full>0` 的调用改前改后完全一样）。
- chunked 内部**不自带 normalize**（`chunked/launch.h:87-103` 的 `chunk_output_config` 只收 `q/k` 指针，无 normalize 开关；
  `prepare_wy_wu_config`/`state_passing_config` 同）⇒ **S_A 的"删掉整个分支"方案会破坏预填（chunked 会收到未归一化的 q/k）**。
  计划书选"把尾块对齐到已发布口径"**是唯一自洽的方向**，本报告确认其必要性。

### 3.3 判据复核：**很可能可达**（我最初的"不可达"判断是错的）

**划分相同**（这是关键，我最初算错成"不同"）：

| 口径 | 调用序列（L=1200） | chunked/recurrent 划分 |
|---|---|---|
| `--prefill-chunk 128` | T=128 ×9（覆盖 [0,1152)）→ T=48（[1152,1200)） | 每次 T=128 ⇒ `T_full=(128/64)*64=128`、tail=0 ⇒ **前 1152 token 全 chunked**；末次 T=48 ⇒ T_full=0 ⇒ **尾 48 token 全 recurrent** |
| `--prefill-chunk 4096` | T=1200 | `T_full=(1200/64)*64=1152` ⇒ [0,1152) chunked；tail=48 recurrent |

⇒ **同一批 token 落到同一条路径上**（18 个 64 块 + 尾 48 token）。

**修③前唯一的口径差**：`gated_delta_net.cpp:255` 的条件是 `normalize_qk && T_full > 0`。
- chunk=4096：T_full=1152>0 ⇒ 全 1200 token 被 `l2norm` 写进 BF16 缓冲（`:258-259`），尾块从该缓冲切片（`:276-277`）
  且 `recurrent_normalize=false`（`:260`）。
- chunk=128：前 9 次调用 T_full=128>0 ⇒ 同样走 BF16 缓冲；**但末次调用 T=48 满足 T_full==0 ⇒ 走 FP32 寄存器归一化**
  （`recurrent.cuh:64-70` + `:617,630` 的 `Normalize=true`）。
⇒ 差异**只有末次调用那 48 个 token 的 q/k 归一化口径**，正是修③ 消掉的东西。

**为什么修③ 之后两侧可以逐位相同**（三条与"调用切分"无关的证据；均为源码级）：
1. `g_cumsum` **每 64 块内重置**：`chunked/prepare_wy_wu.cuh:393-442` —— `const int chunk = blockIdx.x; const int64_t cs =
   chunk * BT;`，扫描 `g_in[cs .. cs+BT)`，`ex_prefix = (lane == 0) ? 0.0f : prev_inc;`，写 `g_cumsum_out[g_row_base + t*H_v]`。
   ⇒ 同一全局 64 块在 chunk=128 与 4096 下得到**同一批 c0/c1**（与调用长度无关）。
2. `l2norm` 契约（`include/ninfer/ops/l2norm.h:9-21`）："Normalizes each logical row over the fastest dimension D=ne[0]"
   ⇒ 每个 (head, token) 独立 ⇒ 对 T=48 与 T=1200 的同一 (head, token) 给出**同一 BF16**。
3. chunked 各块只通过 FP32 `state_in/out` 耦合（`chunked/launch.h:69-85` `state_passing_config`；
   `h_chunk` 的 `chunks` 维只是槽位索引）⇒ 把一个 18 块调用拆成 9 个 2 块调用的**每块算术不变**（FP32 状态精确携带）。

**性能非严格中性**（所以仍建议单独提交、单变量复测）：T∈[2,63]（= 只有 prefill 尾块调用）新增
2 次 `l2norm` 启动 + 2×`kStateDim`×`qk_heads`×T×2 B 的写读（T=48 ⇒ ≈786 KB 往返）。逐请求一次。
**解码稳态（T=1）不走该重载**（`:216` 提前分派到 `launch_recurrent`），**verify 也不走**（走 record/batch_update）
⇒ **decode/verify tok/s 零影响**；只有 prefill 的末尾调用多 2 个 launch。
**精度方向**：修③ 把尾块 q/k 从 FP32 寄存器归一化（`recurrent.cuh:64-70`）降到 **BF16 已发布值**——"以精度换形状无关性"
（不是并行度），**字面不违反**硬约束，但确实是主动降低尾块精度。

---

## 4. 性能影响总表

| 修项 | 结构变化 | 额外 launch | 额外内存/流量 | 结论 |
|---|---|---|---|---|
| ① | 无（1 行，+2 cvt/元素） | 0 | 0 | **中性（arguably 更优：消除 1 次隐式 bf16→fp32 放宽）** |
| ② | verify W∈[4,16] 换 GEMM 路线（MMA→small_t） | **−1**（省掉 `launch_nvfp4_w4a4_quantize`） | 权重不变；激活 +123 KB/层/轮；省掉 w4a4 中间缓冲 | **需实测**（算力比是唯一风险；权重流量不变） |
| ③（可选） | Prefill T∈[2,63] 多走 2 次 l2norm | **+2**（仅尾块，逐请求一次） | 2×2048×T×2 B（T=48 ⇒ 786 KB） | **decode/verify 稳态中性；prefill 尾块微负收益 ⇒ 单独提交、单变量复测** |

---

## 5. 交付物与验证

### 5.1 文件

| 路径 | 内容 |
|---|---|
| `C:\Users\User\Documents\ziqinzhang\_collab\FIX_A_patch.diff` | **修①+修②+修③**（3 文件 4 hunk，2153 B） |
| `C:\Users\User\Documents\ziqinzhang\_collab\FIX_A_patch_no_fix3.diff` | **只修①+修②**（2 文件 2 hunk，1017 B；推荐先落这一份） |

统一 diff 用 `difflib.unified_diff(n=3)` 生成、**LF 行尾**（三份目标文件均为 `C++ source, ASCII text`，无 CRLF；
`.diff` 首行 `--- a/...` 已用 `cat -A` 确认行尾为 `$`）。

### 5.2 dry-run 结果（唯一允许的验证，**未真的改树**）

```
$ patch -p1 -d /home/user/ninfer-fusion --dry-run --verbose < _collab/FIX_A_patch.diff
checking file src/ops/gdn_input_proj/gdn_conv.cuh            -> Hunk #1 succeeded at 96.
checking file src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp -> Hunk #1 succeeded at 40.
checking file src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp
                                                             -> Hunk #1 succeeded at 182.
                                                             -> Hunk #2 succeeded at 252.
done            exit=0
$ patch -p1 -d /home/user/ninfer-fusion --dry-run --verbose < _collab/FIX_A_patch_no_fix3.diff
checking file src/ops/gdn_input_proj/gdn_conv.cuh            -> Hunk #1 succeeded at 96.
checking file src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp -> Hunk #1 succeeded at 40.
done            exit=0
```

`patch` 版本：GNU patch 2.8（`/usr/bin/patch`）。**dry-run 后三份文件 md5 与改前逐字节相同**：

```
3a3b445c91611ea5a8ff2f2c5b62d868  src/ops/gdn_input_proj/gdn_conv.cuh
70c43f510b1b19899f2dcc6813b8d6d8  src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
762f5a36f6e850eed83093290e945956  src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp
```

### 5.3 可复现脚本（留在 `_collab`）

- `_fixa_mkpatch.py`：从三份源文件**现读**、逐条断言"编辑点恰好出现 1 次"、生成上述两份 diff（改树则报错退出）。
- `_fixa_quant.py`：纯 CPU（无 numpy）复算 §1.4/§2.4 的全部数字（E2M1 codebook = `__nv_fp4x2_e2m1` 的
  {0, .5, 1, 1.5, 2, 3, 4, 6}；bf16 用 RNE）。

---

## 6. 残余风险 / 无法确定 / 需主代理实测

### 6.1 需要实测（按优先级）

1. **修② 的性能**（最高优先）：`--spec` 下 W=8/16 的 encode/verify **tok/s 与 round 时间**。
   预期：不变或略优（权重流量不变）。若退化 >3%，把 `:43` 的阈值收到 8（一行）并复测。
   注意"实测"必须先重编（`nvfp4_gdn_snapshot_plan.cpp` 是 .cpp，改动会触发重编；S_A 已证 `.cuh` 也有依赖跟踪）。
2. **修① 的可证伪预测**（S_B P2）：`_ga_check.sh` 的 K=1(W=2) 与 K=7(W=8) 首次偏离位置应**变化/后移**；
   **K=3(W=4) 在修②之前应不变**（W=4 本来 Materialized ⇒ `GdnConvEpilogue` 未被使用）——
   修② 之后 K=3 会**改变**（换到 fused 路），这是**预期行为**，不要误判为回归。
3. **修② 的家族内逐位不变性**（§2.3-2）：结构推断（K 归约序与 ActiveTokens 无关、两种 access 读同一字节）。
   无法离线证；若要证，只需在同一 batch/prompt 下比较 W=2/3/4/8/16 的 `projected`（或直接把 T=1 的 gemv 与
   T=2 的 small_t 在同一 token 上对照，见下条）。
4. **spec==plain 的硬上界**：T=1 走 gemv、T∈[2,16] 走 small_t。**不同 kernel ⇒ 不可能逐位**。
   修①+②后残余预期为"`projected` 的 FP32 末位差（~1e-7）经 bf16 舍入后偶发 ±1 ulp"。
   要达逐位必须再统一 T=1 的 GEMM（S_A §5.2），**这超出本任务**。**建议验收判据改为"同一精度档 + 同一 conv 语义 +
   首次偏离后移/消失"，而不是 IDENTICAL。**
5. **修③ 的判据**（§3.3）：划分已证相同、cumsum 逐块重置、l2norm 逐行独立 ⇒ **`--prefill-chunk 128 vs 4096` 有机会
   IDENTICAL**；唯一未被源码级证明的环节是"chunked 各块只通过 FP32 状态耦合"（§6.2-7）。若实测仍分叉，请把
   分叉位置与 `--prefill-chunk 64`（预期也 IDENTICAL：T=64 ⇒ T_full=64、tail=0，**全部 chunked，连末次调用都不触发
   修③的分支**）对照——这能把"是不是划分问题"一次分开。

### 6.2 我未验证 / 无法确定的边界（诚实声明）

1. **全部为静态阅读 + 离线算术**：未编译、未运行、未 dump 任何张量；§1.4/§2.4 的量级是**同量级估计**（用 N(0,1)
   模拟激活，真实分布是 RMSNorm 后的 hidden，可能改变常数但不会改变数量级）。
2. **我未核对真实 27B artifact 的 `LinearPolicy`**：`AllowA4` 是**源码常量**（`variant.cpp:64,67-70`）推断出的，
   未从 artifact/运行日志确认；若实际是 `A16Only`，修② 变成一行 **no-op**（该分支本来就是 16），修① 仍有效。
   **一行日志（打印 `Fp8GdnConvPlan.schedule` / `Nvfp4GdnConvPlan.schedule`）即可定案**（S_B §7 同样建议）。
3. **`nvfp4_small_t_kernel` 在 T=16 的实际吞吐**未测（§2.5 的风险项）。`kRowsPerWarp` 的取值我未读出
   （`Nvfp4SmallTSchedule` 的第 4 个模板参数语义未逐一确认），因此"权重每行读一次"是按 CTAs=kOutputRows/kRowsPerCta
   的网格推断的。
4. **`gdn_conv.cuh:44-46` 的类注释会变陈旧**：原文 "Projection accumulators stay in the route's existing private precision;
   Publish changes only the side effect after the convolution has consumed that accumulator." —— 修① 后累加器在**被 conv 消费前**
   已被 round 到 bf16。我**故意没改注释**（把 diff 压到 1 行，降低与 B 路 patch 的冲突面）；请协调者在合批时顺手更新。
5. `*.orig` 文件（`gdn_conv.cuh.orig` 等）未纳入比对；我以非 `.orig` 文件为准。
6. **未核对 MTP/DFlash(v1) 路径**是否与 DFlash2 共用同一 plan 函数（`nvfp4_gdn_conv_resolve_plan` 只有 6 个调用点，
   都在 `wrapper/gdn_input_proj.cpp` 与 `nvfp4_gdn_snapshot_plan.cpp`，共享 ⇒ 三个 spec 流派应该都吃到修②）。
7. **修③ 的最后一个未证环节**：我只读了 `chunked/launch.h`（配置/精度）、`prepare_wy_wu.cuh:380-450`（cumsum 逐块重置）
   与 `state_passing.cuh`/`output.cuh` 的**接口与精度**，**未逐行核对**它们的块间耦合。若 `h_chunk`/`v_new` 的
   索引带任何"跨块累计"语义（例如按调用内 chunk 序号索引并复用），拆分调用就不再逐位等价 ⇒ 请以实测为准。

---

## 7. 与 FIX_PLAN_ABCD 的偏差（需要主代理知晓）

1. **修② 不需要改 `nvfp4_gdn_input_w4a4.cu`**（计划书 §2 修② 第二条）：改后 T∈[1,16] **完全不会**进入该文件
   （Materialized 分支下界变成 17，`:44`），所以那里的"激活量化 kernel + M-tile/TMA 表随 T 变"在**小 T 家族内不再起作用**。
   我把"家族内一致"落到 `nvfp4_small_t_kernel` 上并给出了结构级理由（§2.3-2）。**若主代理希望更保守**，可另加
   `if (tokens <= 8)` 之类的保守档，但那会留下 9..16 的 A4 不一致（1e-1 级）⇒ 不建议。
2. **修③ 的判据我最初判为"不可达"，复核后撤回**（§3.3）：`T_full=(128/64)*64=128`（我起初误算成 64），
   所以 chunk=128 与 chunk=4096 的 chunked/recurrent **划分完全相同**，且 `g_cumsum` 逐 64 块重置、`l2norm` 逐行独立
   ⇒ 修③ 有机会拿到 IDENTICAL。**我按计划书语义实现**（"把尾块对齐到已发布口径"，而不是删分支把主路升 FP32 ——
   后者会破坏 chunked，因为 chunked 不自带 normalize，见 §3.2）。**给出但放进单独一份 diff**（`_no_fix3.diff` 是纯①②），
   请主代理决定是否落、并做单变量复测。
3. **修③ 的判据**：见 §3.3 复核 —— 有机会 IDENTICAL；建议判据写"IDENTICAL 或首次偏离显著后移"。
   补充一条最便宜的对照：`--prefill-chunk 64`（T=64 ⇒ T_full=64、tail=0 ⇒ **全部 chunked，末次调用也不触发修③分支**）。
4. 修① 的**收益比计划书描述的更大**：它不只是修"携带"，它是 **plain(T=1) 与 spec(W≥4) 之间 conv 输出的直接不对称**
   （改前 plain 用 24 位 p，spec W≥4 用 9 位 p）⇒ 修① 是 spec-vs-plain 的一阶修复项（2e-3 级），
   而修② 是更高一阶（1e-1 级）。**建议落地顺序：②（收益最大）→ ① → 复测 → 再考虑 ③。**
