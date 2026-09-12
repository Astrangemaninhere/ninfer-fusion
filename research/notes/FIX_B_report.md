# FIX_B 报告（B 路，2026-09-11）

证据范围：只读源码（`sed -n` 小窗口 + `grep -n` 定点，命令全部先落 `.sh` 再经 `wsl.exe bash -c` 执行）、
离线 CPU 量化（`_collab/_fb/offline.py`，numpy，未编译/未跑引擎/未用 GPU）。
构建树 md5 在打补丁前后**未变**（见 §5）。

产物：
- `_collab/FIX_B_patch.diff`（3 文件 / 4 hunk）
- `_collab/_fb/mkpatch.py`（生成 diff，不改树）、`_collab/_fb/verify.sh`（dry-run + 复制树实跑 + md5 校验）、`_collab/_fb/offline.py`（量化证据）

---

## 0. 结论先行

| 项 | 改动 | 性能 | 数值效果（有据） |
|---|---|---|---|
| **修①** | `gdn_conv.cuh:99` 一行：`projected[token]` → `__bfloat162float(__float2bfloat16_rn(projected[token]))` | **中性**（每 (token,row) 多 1 条 f32→bf16 RNE，4-tap 循环内） | fused conv 与 materialized conv **逐表达式相同** ⇒ 修后两者对该 p **逐位等价**；状态窗口/记录平面**逐位不变**（publish 本来就存 `bf16(p)`） |
| **修②** | 语义核心**已由另一路落入树中**（`nvfp4_gdn_snapshot_plan.cpp:43` 的 `tokens <= 3 → <= 16`，mtime **15:13:52**，见 §2.0）；我的 hunk 是同一谓词的**理由注释**（不revert 已落地改动） | —— | 消除 verify 侧**一阶偏差**：A4 激活量化相对误差 RMS **9.4e-2** vs bf16 **1.7e-3**（离线实测，57×，见 §4D）⇒ W∈[4,16] 从"另一个精度档+另一套 conv"回到与 plain 同档同代码 |
| **修③（可选）** | `gated_delta_net.cpp:182-195` + `:255`：`normalize_qk` 恒走 BF16 归一化缓冲（尾块对齐已发布口径） | **中性**（+2 个小 l2norm launch；arena 增量见 §6-4） | 把 `T_full==0` 与 `T_full>0` 的尾块统一到同一发布精度；**但不会**让 chunk128 vs chunk4096 逐位一致（残余来自 attention/MLP/lm_head 的 T 形状 kernel，S_A §7.4-4 已预警） |

**修① 的关键自洽性论证（不依赖实验）**：`GdnConvEpilogue::store`（fused）与
`nvfp4_gdn_conv_post_kernel`（materialized）的卷积段是**同一串 fmaf**：
`fmaf(w0,s0,0) → fmaf(w1,s1,·) → fmaf(w2,s2,·) → fmaf(w3,p,·)`，之后同样是 `bf16_rn(silu(conv))`。
材料化路的 `p` 来自 **BF16 缓冲**（`gdn_conv.cuh` 同族：`nvfp4_gdn_snapshot_post.cu:96`、
`gdn_projected_conv.cu:53`）；fused 路的 `p` 是 FP32 GEMM 累加器。修① 令两者同源 ⇒
**修后 fused conv 与 materialized conv 对相同 `projected` 的输入逐位相同**（不是 1e-7，是 0）。
这条把 S_B 的 D1（~2e-3）与 D2（~1e-7）**分开**：D2 只在**非 conv** 的 GEMM 归约树上残留。

**修① 对 plain decode 也生效（重要，纠正 S_A §4.2 的过度概括）**：S_A 说"`Tokens==1` ⇒ `:118` 补丁是死代码"，
这对 `:118`（s2 进位）成立，但对**修①不成立**：`projected[token]` 直接进 `fmaf(w3,p,·)` 再出 `silu`，
所以 `GdnConvEpilogue<1>`（plain decode / `DecodeFusedA16`）的输出**会变**。即修① 同时改动 plain 基线，
方向正确（prefill 的 `causal_conv1d_silu` 读的就是 BF16 列）。

---

## 1. 修① 证据（确切行 + 改前/改后原文）

`src/ops/gdn_input_proj/gdn_conv.cuh`

- `:99` 改前：`            const float p              = projected[token];`
- `:99` 改后：`            const float p              = __bfloat162float(__float2bfloat16_rn(projected[token]));`
- `:103` `conv = fmaf(w3, p, conv);`（不变）
- `:113` `publish.publish(token, batch_row, row, s1, s2, p);`（不变，语义中立，见下）
- `:118` `s2 = __bfloat162float(__float2bfloat16_rn(p));`（今天已落地的补丁，保持）

**发布侧逐位不变（可证）**：两个真实 Publish 都是
`:21-23` `state_write[..] = __float2bfloat16_rn(p)`（`SnapshotHistoryPublish`）与
`:35` `record[column*channels+row] = __float2bfloat16_rn(p)`（`RecordColumnPublish`）。
`bf16` 幂等 ⇒ 改后传入的 `p` 已是 bf16 值，**写出的位完全相同**；`:118` 的 s2 亦等于改前的
`bf16(p)`。⇒ 修① 只改 `silu(conv)` 的输出，**不改任何状态/记录**。

**触发面（确切代码行判定）**：所有 fused 路线共用该模板，无例外：
`nvfp4_gdn_snapshot_decode.cu:23`（`make_gdn_conv_output<1>`，plain decode/`DecodeFusedA16`）、
`nvfp4_gdn_snapshot_small_t.cu:33-40`（`SmallTFusedA16`，W=2..16）、
`fp8/fp8_gdn_conv_fused.cu:41,82`、`q4_q5/q4_q5_gdn_input_conv_snapshot.cu:40-312`、
`w8/w8_gdn_input_decode.cu:13,28`、`w8/w8_gdn_input_gemm_splitk.cu:33`。
**不**走该代码的三条（已核实 `p` 来自 BF16 缓冲，修① 对它们是 no-op）：
`nvfp4_gdn_snapshot_post.cu:96`、`gdn_projected_conv.cu:53`、
prefill 的 `causal_conv1d_silu`（`text_context_impl.h:1135`）。

**量级（离线实测，§4C）**：`|p − bf16(p)|/|p|` 均值 **1.409e-3**、最大 3.891e-3
（bf16 是 8 位有效位：典型半 ulp `2^-9 = 1.953e-3`，紧邻 binade 边界时 `2^-8 = 3.906e-3`）。
⇒ 与 S_B 的 "~2e-3" **同量级且已量化**。输出的绝对改变量有确定上界
`|Δconv| ≤ |w3|·2^-8·|p|`；我的尺度模型（p,s ~ N(0,0.6)、w ~ bf16(U(-0.3,0.3))、4e6 列）给出：
发布的 bf16 q/k/v 在 **14.84%** 的列上改变，其中 **97.1%** 恰好是 1 个 bf16 步，
`|Δout|` 均值 2.3e-4 / 最大 3.9e-3。

---

## 2. 修② 证据

### 2.0 并发事实（先读这条）

我 15:05 读到该文件时 `:43` 是 `if (tokens <= 3) { … SmallTFusedA16 }`（md5 `70c43f51…`）；
**15:13:52** 该文件被**另一路**改写（md5 `0b351a16…`），`:43` 变成 `if (tokens <= 16)`，
**语义与计划要求完全一致**；此外树内 `gdn_conv.cuh`（md5 `3a3b445c…`）与
`gated_delta_net.cpp`（md5 `762f5a36…`）**未被动过**（即修① 与修③ 仍空缺）。
⇒ 我**刻意**把我的 hunk 2 退化为"理由注释"（`@@ -34,6 +34,11 @@`，只插入 5 行注释），
**不 revert** 另一路的落地改动；这样即使两路 patch 都被应用也不冲突。
修② 的**实质贡献**因此从"改一行"转为**验证**：谓词穷举表（§4A）、可达域（§2.2）、
以及"w4a4 表在小 T 内本来就一致"的查证（§2.3）。若主代理希望采纳我的**结构形式**
（把 `tokens == 1` / `tokens <= 16` 提到 policy 判断之前，令两个 policy 显式共用一张表），
可自行搬动该三行；我从 diff 里拿掉它，避免与已落地改动churn。

### 2.1 分档点原文

`src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp`（当前树）
```cpp
:36    if (batch_size > 1) { return {Nvfp4GdnConvScheduleId::Materialized}; }
:37    if (policy == LinearPolicy::A16Only) {
:38        if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
:39        if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
:40        throw std::invalid_argument("nvfp4 gdn conv A16 is registered only through T=16");
:41    }
:42    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
:43    if (tokens <= 16) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }  // ← 15:13:52 由 <= 3 改成
:44    return {Nvfp4GdnConvScheduleId::Materialized};
```
两 policy 现在在小 T 区间**逐格相同**（`:38-39` ≡ `:42-43`）；唯一差别是 A16 在 T>16 时 `throw`
（`:40`）而 A4 materialize（`:44`）。

### 2.2 触发谓词与可达域（离线穷举，§4A）
`(policy, tokens, batch_size)` 的纯函数。after−before **只在 13 个格点变化**，全部是
`policy=A4 ∧ batch=1 ∧ tokens ∈ [4,16]`：`Materialized → SmallTFusedA16`。
其余格点（含 A16 全表、batch>1 全表、B=1 的 T=1..3 与 T≥17）**逐格不变**。

**这些格点确实可达**：record（verify）域由
`src/ops/wrapper/gdn_input_proj.cpp:78-89`（`require_record_input`）限定 `2 ≤ W ≤ 16`、`B ≤ 8`；
snapshot 域 `:66-76`（`require_snapshot_input`）`B=1` 时 W 无上界。
⇒ **本 artifact（`kNvfp4TextPolicy = AllowA4`，`src/targets/qwen3_6_27b/impl/variant.cpp:64`）
的所有 NVFP4 单 batch record 调用（W ∈ [2,16]）在修后全部是 `SmallTFusedA16`**：
`wrapper:610-611` 解析、`:621-626` 走 `nvfp4_gdn_record_small_t_launch`，
因此 `:627-632` 的 Materialized record 分支与 `:903` 的 record 容量分支对 NVFP4/B=1 变为**死路**。

**关于计划要求的"确认 `nvfp4_gdn_input_w4a4.cu:40-58` 一致"**：查证结果——**小 T 家族内本来就一致，
而且修② 之后 T≤16 根本不再进入 w4a4**：
- 激活量化布局：`nvfp4_w4a4_plan.h:56-60` `nvfp4_w4a4_tma_route(tokens) = tokens >= 1024 && tokens % 256 == 0`
  ⇒ 对 T≤16 恒为 `false`，**同一个量化 kernel 变体**（`nvfp4_gdn_input_w4a4.cu:40`）。
- M-tile：`:48-51` `tokens <= 64 → M32N64`（唯一覆盖 T≤16 的分支）；TMA 分支 `:42` 需 `tokens >= 1024`
  ⇒ 小 T 内**唯一** M-tile。
- 所以"不一致就统一"这一条**无改动对象**；修② 之后 T∈[1,16] 的 conv 路径只剩
  `DecodeFusedA16`(T=1) 与 `SmallTFusedA16`(T=2..16)，两者都**不量化激活**（直接读 BF16 `x`：
  `nvfp4_gdn_snapshot_decode.cu:21`、`nvfp4_gdn_snapshot_small_t.cu:36`），精度档统一为 A16。

**无需新 kernel**：`nvfp4_gdn_snapshot_small_t.cu:70-84` 的 launcher 表（snapshot + record）
覆盖 `ActiveTokens = kNvfp4FirstSmallT(2) .. 16`（`nvfp4_config.h:196-197`），
且 `Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, T>` 对 T=2..16 已全部实例化
（`nvfp4_config.h:220-232`）⇒ 修② **不改变目标文件里的 kernel 集合，只改分派**（编译期风险≈0）。

**已存在同配置**：W=2、3 今天就已经是 `SmallTFusedA16`，且 record 容量今天就已经走
`:902 if (maximum_plan.schedule == SmallTFusedA16) return 0;` ⇒
"容量=0 + SmallT record"这条组合是**当前活配置**，修② 只是把它扩到 W=4..16，**不引入新行为类**。

---

## 3. 性能影响论证

修①：单条 `__float2bfloat16_rn` + 1 条 cvt，位于原本就有 `silu`（含 `expf`）与一次
`__float2bfloat16_rn` 的循环体内 ⇒ 结构、访存、并行度、占用率全不变。**中性**。

修②：把 W∈[4,16] 的 GDN 输入投影从 `W4A4 (mma_nvfp4_e4m3 张量核 + 激活 FP4 量化 + 独立 post conv)`
换到 `A16 (nvfp4_small_t_kernel，SIMT fmaf + 融合 conv 尾声)`。要点：

1. **两侧读同一份权重字节**：该投影 16384×5120 NVFP4 ≈ **42 MB**（4bit+scale）。以
   `(16384·5120·T)` 个 MAC 计，W4A4 的 mma 时间在 T≤16 时 ~1.6 µs（按 1.68 PFLOPS fp4 估），
   远低于取权重的时间 ⇒ **W4A4 侧是带宽下界 ~23 µs（1.79 TB/s）**。
2. A16 侧：T=16 时 1.34 G MAC ⇒ 672 M 条 fma 指令 / (170 SM × 128 lane × 2.5 GHz ≈ 54 T fma/s)
   ≈ **12 µs** 计算 + 权重取数 23 µs ⇒ 仍**藏在同一带宽屋顶下**（激活再读的 L2 流量是次要项）。
3. 上界估计（最坏情形把 A16 侧算到 2× 带宽下界）：**每个 GDN 层 +25 µs**；
   按 32 层、W=16 的 round 时长 ~50–110 ms 估 ⇒ **+0.7~1.6 ms ≈ 1–3%**。
4. 修② **不会**新增 kernel 实例化（见 §2），也不会改变 decode（T=1）的任何路径。
5. 结构性证据：这条 A16 融合路径在 W=2、3 上**已经在跑**（S_A 表），因此不是"未验证的新路径"。

结论：**性能中性到最多低个位数百分比**，但**必须由主代理用解码 tok/s 实测钉死**（§7）。

修③：`normalize_qk` 恒走 BF16 缓冲 ⇒ 多 2 个 `l2norm` launch
（规模 `128 × qk_heads × T`，T<64，27B 下 qk_heads=16 ⇒ ~13 万元素/个，个位数 µs），
同时从 recurrent kernel 里去掉了寄存器归一化（少一次 `rsqrtf` 链）。**中性**。

---

## 4. 离线（CPU）可算的数字

`_collab/_fb/offline.py`（numpy，无 GPU）：

### A. schedule id 穷举（修② 的谓词表）
before/after 共 **13 个格点变化**，全部 `A4 ∧ B=1 ∧ T∈[4,16]` → `Materialized → SmallTFusedA16`。
**因此不需要在引擎里打日志来定案**。若确实想加一行开发期日志，位置是
`nvfp4_gdn_snapshot_plan.cpp:68`（dispatch 的 switch）或调用点 `wrapper/gdn_input_proj.cpp:611`
（record）/`:451`（snapshot）——但注意 `nvfp4_gdn_conv_resolve_plan` 是纯函数，
日志只能复述本表。

### B. record 域（`require_record_input` 的 `kMaximumWidth = 16`）⇒ 修后**全部** W∈[2,16] 为 SmallT。

### C. 修① 的 bf16 舍入量化（4e6 列蒙特卡洛）
| 量 | 值 |
|---|---|
| `\|p − bf16(p)\|/\|p\|` 均值 / 最大 | **1.409e-3** / 3.891e-3 |
| 理论（8 位有效位） | 半 ulp `2^-9 = 1.953e-3`；紧邻 binade 边界 `2^-8 = 3.906e-3` |
| 发布 bf16 q/k/v 改变的列比例 | **14.84%** |
| 其中恰好差 1 个 bf16 步 | **97.1%** |
| `\|Δout\|` 均值 / 最大 | 2.32e-4 / 3.906e-3 |
| 确定上界 | `\|Δconv\| ≤ \|w3\|·2^-8·\|p\|` |

**为什么是 1e-3 级**：舍入发生在**一次** f32→bf16（8 位有效位）上，且只经 1 个 tap 权重 `w3` 传播；
不存在二次累加/链式放大。状态与记录平面被证明**逐位不变** ⇒ 该改动是"窄口径、可预测"的。

### D. 修② 的精度档悬崖（同一个模型）
| 激活量化（a ~ N(0,1)，per-16 块，e2m1 码 + max-abs fp8 块 scale） | 值 |
|---|---|
| `RMS(\|a − quant(a)\|)/RMS(a)` NVFP4 / bf16 | **9.417e-2** / 1.657e-3（**56.7×**） |
| 元素相对误差中位数 NVFP4 / bf16 | 1.019e-1 / 1.4e-3 |
| K=5120 随机符号点积的 `\|Δproj\|/\|proj\|` 中位 NVFP4 / bf16 | **9.63e-2** / 1.71e-3（p90: 5.81e-1 / 1.08e-2） |

⇒ A4 激活路线的偏差是**一阶的（~1e-1）**，比修① 要消除的 bf16 类（~1e-3）大 **~2 个数量级**；
这正是 S_A 观察到的 W=3→4"台阶"而非渐变的量与机制：**任何下游 conv 修补都无法抵消激活被量化成 FP4**。
修① 负责把 W≤3 及 plain decode 的 conv 语义对齐，修② 负责把 W∈[4,16] 从 A4 档拉回 A16 档，
两件缺一不可。

---

## 5. diff 与 dry-run 结果

`FIX_B_patch.diff`（69 行，4 hunk / 3 文件）：
- `src/ops/gdn_input_proj/gdn_conv.cuh` — `@@ -96,7 +96,11 @@`：**修①（唯一实质代码改动）**
- `src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp` — `@@ -34,6 +34,11 @@`：5 行注释（§2.0，不 revert 已在树的 `<= 16`）
- `src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp` — `@@ -180,17 +180,23 @@` + `@@ -252,7 +258,7 @@`：**修③（可选，删掉该文件的 hunk 即跳过）**

```
$ cd /home/user/ninfer-fusion && patch -p1 --dry-run < _collab/FIX_B_patch.diff
checking file src/ops/gdn_input_proj/gdn_conv.cuh
checking file src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
checking file src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp
dry-run rc=0
```
- 打补丁前后三个文件的 md5 **完全一致**
  （`3a3b445c…` / `0b351a16…` / `762f5a36…`，`verify.sh` 记录），即 dry-run 未改树；
  另在 `/tmp/fixb/dryrun` 的**复制树**上实跑 `patch -p1`（rc=0）并复查了结果文本。
- **注意**：本 diff 是针对 **15:13:52 之后的树**生成的（即已含另一路的 `<= 16`）。
  若主代理在 15:14 之后又改动这三个文件中的任一个，请以「先 application-order 后顺序」重取，
  或直接只取 hunk 1（修①）——它不依赖另外两个文件。
- 行尾：三文件均为 **LF**（`file` + `cat -A` 核对），diff 中无 CR。
- 二次 dry-run 报 "Reversed (or previously applied) patch detected" ⇒ 幂等性正常。
- 修③ 只涉及 `gated_delta_net.cpp` **一个文件**的两个 hunk ⇒ **只应用前两个文件即跳过修③**。

---

## 6. 残余风险

1. **spec≠plain 不可能逐位**（预期内）：T=1 走 `nvfp4_gemv_kernel`（decode schedule
   `Nvfp4GemvSchedule<8,2,16,4,StagedRaw,Default,2>`，`nvfp4_config.h:190-194`），T=W 走
   `nvfp4_small_t_kernel`（`Nvfp4SmallTSchedule<…>`，`nvfp4_config.h:220-232`），
   两者的归约树不同（`nvfp4_gemv.cuh:239-242` vs `nvfp4_small_t.cuh:273-276` 的
   `kAccumulatorChains` + `warp_reduce_sum`）⇒ `projected` 仍有 ~1e-7 级差（S_B 的 D2 同族）。
   修①+② 把"精度档/算法/conv 语义"三层差异清零，**只剩**这一类。验收请用 `_df2_blame2.py`
   的逐列剖面与首次偏离位置，而不是"逐位相等"（S_A §5.2 已给同样的退让建议）。
2. **batch>1 仍不自洽**：`B>1` 走 `compose_batched_snapshot/record`（`wrapper:456-462`、`:701-707`）
   ⇒ 激活仍经 `nvfp4_gdn_input_dispatch(…, AllowA4)` = W4A4，conv 仍走逐列 post。
   本次**不动**。若要修，得让 compose 的 λ 也走 A16（`nvfp4_gdn_input_plan.cpp:17-22`
   的 route 对 A16Only 恒 A16），改动面比修② 大。这是 S_A §5.3-3 的 B 维度问题的正主。
3. **悬崖只上移到 T=17**：T∈[17,64] 仍是 W4A4。`kLaunchers` 已注册到 32，`kNvfp4LastSmallT=32`；
   但 `require_record_input` 的 `kMaximumWidth=16` 把 record 挡住，所以本 artifact 的 verify 到不了。
   若主代理在 K 扫描里看到 W=17 处再现台阶，才值得把上限一起放。
4. **修③ 的 arena 增量**：`allocate_chunked_workspace` 现在对 `0 < tokens < 64 ∧ normalize_qk`
   也分配两张 BF16 归一化缓冲（`2 × 128 × qk_heads × tokens × 2 B`；27B 的 qk_heads=16、
   T=63 ⇒ **~516 KB**）。容量函数与调用点同源，故一致；但如果某个 workspace plan 的
   `max_tokens < 64` 且其 arena 是**外部硬算**的，需要主代理扫一眼（我未逐处核对全部 plan 构造点）。
5. **修③ 无单测覆盖**：`tests/ops/test_gated_delta_net.cpp` 的 chunked in/out 用例只有
   `inplace_case(T=1)` 与 `inplace_case(T=65)`（`:461,466`），T=63 的用例走
   `gated_delta_net_batch_update`（`:462` 的 `distinct_state_case`，`:340`）⇒
   修③ **不会**让现有单测变红，**也就意味着它没被单测保护**。
6. **验收基线的读法**：修① 改动 plain decode 的 q/k/v（14.8% 列 × ≤1 bf16 步）、
   修② 改动 W∈[4,16] verify 的整条投影 ⇒ `_ga_check.sh` / `_ga_ksweep.sh` 的**全部** hash
   都会变。**"和修前的 hash 不同"不是回归**；判据应是"修后首次偏离位置后移/消失 + 接受率不降 +
   tok/s 不降"。同时 K=3(W=4)/K=7(W=8) 的"逐位不变"现象**必然消失**——这正是修② 生效的证据（S_A §4.2-3 的预言反向）。
7. `gdn_conv.cuh.orig` 仍在树里（`:113-118` 显示原 `s2 = p`），本次未动、也不该动。

---

## 7. 无法确定 / 需要主代理实测（清单）

0. **归属与时间戳（先做）**：修② 的 `tokens <= 16` **已在树中**（`nvfp4_gdn_snapshot_plan.cpp`
   mtime `2026-09-11 15:13:52`，由另一路落入）。因此**任何 15:13:52 之后**测得的 tok/s /
   接受率 / `_ga_check` 结果**已经包含修②**；而修①（`gdn_conv.cuh`）**尚未**落地
   （md5 `3a3b445c…` 仍是旧内容）。归因前请先核对测量与改动的先后，否则会把修② 的收益算到别处、
   或把修① 的缺失当成"修② 无效"。

1. **解码 tok/s**（性能硬约束的唯一裁决）：修② 后 W=16 的 verify 段。我给的界是 **+0~3%**（§3）。
2. **`_ga_check.sh`（zh + num）首次偏离位置**：修①+② 的联合效果。我的可证伪预测：
   - 因为修① 同时改 plain 与 spec 两侧的 conv 语义、修② 又消掉 W≥4 的 A4 档，
     **首次偏离应显著后移**（K=1/K=3/K=7 都应动；不要再期望 K=3 "不变"）。
3. **接受率曲线 vs W**（`_verify_df2head.sh` / `_df2_blame2.py`）：用于验证 §4D 的 1e-1→0 一步。
   若 W=4..16 的接受率出现台阶式提升 ⇒ 修② 命中；若无变化而 W=3 与 plain 仍不等 ⇒
   残余主因转到 selector（S_D）与非 GDN 的 T 形状 kernel。
4. **修③ 是否采用**：若采用，跑 `_align128_ab.sh`（**必须非 128 对齐 prompt**）。
   预期是"首 token 分歧变小/后移而**不为 0**"；若期望 IDENTICAL，则修③ 单独不够
   （还需 attention/MLP/lm_head 的 T 形状统一，属计划外）。
5. **是否要打 schedule id 日志**：我认为**不需要**（§4A 已给出穷举表）；若要，按 §4A 的位置加。
6. 我**没有**读 25 GB 的 `.ninfer` artifact（无法在此离线读权重张量统计），
   §4C/4D 的分布是**尺度模型**而非该 ckpt 实测；若主代理能 dump 出某个 GDN 层
   `projected` / conv 权重 `w3` 的真实直方图，`|Δout| ≤ |w3|·2^-8·|p|` 的上界可直接套用。
7. **未编译**：修①/② 是类型安全的（`__float2bfloat16_rn`/`__bfloat162float` 均在
   `<cuda_bf16.h>`，`gdn_conv.cuh:5` 已包含；`nvfp4_gdn_snapshot_plan.cpp` 只搬动语句、
   无新符号），但 `clang-format`(ColumnLimit=100) 我只做了**行宽检查**（≤96/≤99），
   未跑 formatter——如需 CI 格式门，请主代理 `clang-format -i` 后复核。
