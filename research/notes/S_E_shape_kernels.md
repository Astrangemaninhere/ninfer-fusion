# S_E — 形状相关（按 T 分派）数值分叉源清单

树：`/home/user/ninfer-fusion`（只读分析，未改源码、未编译、未跑引擎）
约定：`M` = 该 prompt 的 token 数。`--prefill-chunk 128` → 每次调用 T=128，末块 T = M%128（M=1200 时 T=48；M=1800 时 T=8）；
`--prefill-chunk 4096` → 单次调用 T=M（1200 或 1800）。下文用 **T∈{48,128,1200}** 代表前者与后者。
模型判定：下文以 **qwen3_6_27b（dense, NVFP4 权重）** 为主假设（与主代理此前 nvfp4 GDN 修补一致），
MoE 分支（qwen3_6_35b_a3b / groupwise-int）单独标注，因为它会改变排序。

---

## 0. 结论先行

| 排名 | 分叉源 | 判定 | 触发条件（本用例） | 数值量级 |
|---|---|---|---|---|
| 1（MoE 模型则为 #1） | MoE 专家 GEMM 的 `adaptive` 分档 `tokens>=47 && tokens<=adaptive_last` | 确定会分叉（仅当 routed_down=Q5/Q6） | T=48 → adaptive（走 decode small-t 专家核）；T=128/1200 → grouped prefill | 整个专家 GEMM 家族不同 |
| 1（dense 模型） | MLP gate-up：`TmaFusedW4A4` epilogue **缺 bf16 回舍** vs baseline/MMA-fused 有回舍 | 确定会分叉（值级，非调度级） | T=1200 会切 head=1024 走 TMA；T=128 永不走 TMA | ~2^-9 ×2 每层激活 |
| 2 | GDN gating proj 的 SplitK 随 T 变档（`cols>1024` → Split4，否则 Split8/16） | 确定会分叉（fp32 累加分组不同） | T=1200 → Split4；T=48/128 → Split8（27B）/Split16（35B） | ~1e-7 相对，但喂进 exp() |
| 3 | MoE `wide_plan = tokens>=768` → route_job_bn 32/64 | 可能（无 K 拆分，需确认 grouped 归约顺序） | T=1200 wide；T=48/128 窄 | 小 |
| 4 | 其它权重档（w8 / q5 / fp8）的按 T 路由表（含 `SplitKMma*`） | 确定会分叉 —— 但**仅在权重档不是 nvfp4 时**才被走到 | 本 nvfp4 artifact 不走 | —— |
| — | nvfp4 MMA 家族的 6 档 schedule 阶梯 | **不会**（见 §2.1，同一 token 逐位一致） | 全部命中不同 schedule | 0 |
| — | rope / causal_conv1d / KV fill(`tokens>=128`) / 注意力路由 / rmsnorm / lm_head | **不会**（见 §2.2–2.7） | 全部命中不同 kernel | 0 |

---

## 1. 逐个分派点：`file:line` + 谓词原文 + T 解析

### 1.1 nvfp4 线性家族（同一 MMA kernel，只有 schedule 变）

对所有 `nvfp4_w4a4_mma_kernel<Geometry, Schedule>` 使用者，T 解析如下（六档完全一致）：

```
T <= 64   -> M32N64            T <= 192 -> M64N128
T <= 96   -> M32N128           T <= 384 -> M128N128Resident
T <= 128  -> M128N128Pipelined T <= 512 -> M128N128Pipelined(*)
else      -> M128N128Resident
tokens >= 1024 && tokens % 256 == 0 -> TMA 路线
```

| 文件:行 | 谓词原文 |
|---|---|
| `src/ops/linear/nvfp4/nvfp4_w4a4_plan.h:56` | `inline constexpr std::int32_t kNvfp4TmaBlockM = 256;` |
| `src/ops/linear/nvfp4/nvfp4_w4a4_plan.h:58-60` | `return tokens >= 1024 && (tokens % kNvfp4TmaBlockM) == 0;` |
| `linear/nvfp4/nvfp4_w4a4.cu:64` | `if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {` |
| `linear/nvfp4/nvfp4_w4a4.cu:71,73,75,81,83,85` | `tokens <= 64 / 96 / 128 / 192 / 384 / 512` |
| `linear/nvfp4/nvfp4_w4a4.cu:76-80` | `if constexpr (kResidualGeometry) { launch_gemm<Geometry, M32N128>(...)` ← T≤128 时 residual 几何**换档**成 M32N128 |
| `linear/nvfp4/nvfp4_w4a4.cu:86-90` | `if constexpr (Geometry::kOutputRows == Nvfp4GdnInputGeometry::kOutputRows)` ← T≤512 时 GDN 几何换档 |
| `attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu:88,95,97,99,101,103,105` | `tokens >= 1024 && (tokens % kTmaBlockM) == 0` / `tokens <= 64 / 96 / 128 / 192 / 384 / 512` |
| `linear_add/nvfp4/nvfp4_linear_add_w4a4.cu:62,40,42,44,46,48` | 同上（无 TMA 分支以外的一致阶梯） |
| `gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu:42,48,50,52,54` | `tokens >= 1024 && (tokens % kTmaBlockM) == 0` / `tokens <= 64 / 96 / 128 / 192`（else → M128N128Resident） |

T 解析（本用例）：

| 算子 | T=48 | T=128 | T=1200 |
|---|---|---|---|
| `attn_input_proj`(q/k/gate/v) | M32N64 | M128N128Pipelined | M128N128Resident（1200%256≠0 → 非 TMA） |
| `gdn_input_proj`(qkv/z) | M32N64 | M128N128Pipelined | M128N128Resident |
| `linear_add`(o_proj K=6144 / down K=17408) | M32N64 | M32N128 | M128N128Resident |
| `linear`(MTP/其它) | M32N64 | M128N128Pipelined | M128N128Resident |

A16/W4A4 路由阈值（在本用例**都不触发**，T≥48 全 W4A4）：
`linear/nvfp4/nvfp4_dispatch.cpp:32,36,39` = `tokens >= 4 / 5 / 8`；
`attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp:24` = `tokens >= 4 ? W4A4 : A16`；
`linear_add/nvfp4/nvfp4_linear_add_plan.cpp:26-27` = `first_w4a4 = input_rows == 6144 ? 7 : 8;`
（`Muse*` 几何恒 A16，`linear/nvfp4/nvfp4_dispatch.cpp:40-45`。）

### 1.2 MLP gate-up（nvfp4_linear_swiglu）——**唯一的 nvfp4 值级分叉**

`src/ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp`：

```
:33  if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
:34  if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
:37  if (tokens == 1 && std::getenv("NINFER_T1_W4A4") == nullptr) { ... DecodeFusedA16; }
:40  if (tokens <= 4 && std::getenv("NINFER_T1_W4A4") == nullptr) { ... SmallTFusedA16; }
:46  if (tokens <= 48) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
:47  if (tokens >= kTmaBlockM && (tokens % kTmaBlockM) == 0) { return ...TmaFusedW4A4; }   // kTmaBlockM=256
:50  return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
:128 if (route == Nvfp4LinearSwiGluRoute::LinearW4A4Post && tokens >= 2 * kTmaBlockM) {
:133     const std::int32_t head = tokens & ~(kTmaBlockM - 1);   const std::int32_t tail = tokens - head;
:141     launch_nvfp4_linear_swiglu_w4a4_tma(... head ...);      // head 走 TMA
:151     linear(tail_x, weight, scratch.projected, LinearPolicy::AllowA4, ...); silu_mul(...);  // tail 走 baseline
```

T 解析：

| T | route | 内核 | gate/up 是否先回舍 bf16 |
|---|---|---|---|
| 48 | `FusedW4A4` | `nvfp4_linear_swiglu_w4a4.cu:94` `launch<M48N64>`，`M48N64 = Nvfp4W4a4MmaSchedule<48,64,256,3,4,2,2>` | **是**（见 §2.1 论证：等价 baseline） |
| 128 | `LinearW4A4Post` | `linear()`(M128N128Pipelined) + `silu_mul` | **是**（`linear` 落盘 bf16） |
| 1200 | `LinearW4A4Post` 分裂 | head=1024 → `nvfp4_linear_swiglu_w4a4_tma.cuh`；tail=176 → `linear()`(T≤192→M64N128)+`silu_mul` | **head 否 / tail 是** |
| 256·k (k≥1) | `TmaFusedW4A4` | 全走 TMA | **否** |

关键代码事实：TMA 融合 epilogue 直接在 fp32 累加器上做 silu——
`nvfp4_linear_swiglu_w4a4_tma.cuh:242-245`
`*destination0 = __floats2bfloat162_rn(silu(gate[0] * alpha) * (up[0] * alpha), silu(gate[1] * alpha) * (up[1] * alpha));`
（`gate`/`up` 是 `accumulators[...]`，未经过任何 bf16 转换。）
而 baseline 与 MMA 融合两条路都先在 bf16 上做 silu（§2.1）。→ **值级分叉，确定**。

### 1.3 GDN gating proj（bf16）——SplitK 随 T 变档

`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:29-48`：

```
k27Routes = { {1,1}->GemvPairedRows, {2,8}->SmallTSplit10, {9,1024}->MmaCooperativeSplit8,
              {1025,2048}->MmaCooperativeSplit4, {2049,4096}->MmaCooperativeSplit2, {4097,∞}->MmaUnsplit }
k35Routes = { {1,127}->MmaCooperativeSplit16, {128,1024}->Split8, {1025,2048}->Split4, ... }
:339-343  variant = (cols % mma_tile_cols) == 0 ? Full : Predicated
:343      const std::int32_t split_k = schedule_split_k(schedule);
:344-345  split_k > 1 ? checked_partial_bytes(problem.heads, split_k, problem.cols) : 0
```

`bf16_gdn_gating_proj_kernels.cu:258-303`：`template <class Geometry, int SplitK, ...>`，
`const dim3 grid(div_up(t, kBlockN), Geometry::kHeads/kBf16GdnBlockM, SplitK);` 且 `SplitK>1` 走 cooperative launch；
`kernels.cu:279 if constexpr (SplitK > 1)`。partial 缓冲 `split * tokens * 2*heads` 个 float（plan.cpp:203-214）
→ 这是**沿归约轴（K）的拆分**，拆分块数随 T 变化 → fp32 求和分组不同。

T 解析：T=48/128 → 27B `Split8`（35B `Split16`/`Split8`）；T=1200 → **`Split4`**。
→ **值级分叉，确定**。g/beta 的差约 1e-7 相对，但 g 进入 `exp()`/softplus、并由 GDN 递推放大。

### 1.4 MoE（仅 qwen3_6_35b_a3b 类模型）

`src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu`：

```
:1147 route_tiles = (tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens
:1150 const int adaptive_last = weights.routed_down.qtype == QType::Q5G64_F16S   ? 51
:1151                           : weights.routed_down.qtype == QType::Q6G64_F16S ? 52 : 0;
:1153 const bool adaptive     = tokens >= 47 && tokens <= adaptive_last;
:1165 const bool wide_plan   = tokens >= kSparseMoePrefillWideMin;   // 768, sparse_moe_prefill.h:20
:1166 const int route_job_bn = wide_plan ? 64 : 32;
:1172 if (adaptive) { sparse_moe_decode_launch_d3_small_t(... Paths3 ...); sparse_moe_decode_launch_d4_small_t(... Rows4 ...); }
```

前置闸门 `src/ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp:11-28`：
`prefill_min_tokens` 只在 `routed_gate_up == Q4G64_F16S`（down Q5/Q6 → 47）或 `W8G32/W8G32`（→20）时非 0；
否则 `sparse_moe_uses_prefill` 为假 → 整个 MoE 走 decode 路径。另有 `sparse_moe/small_t/sparse_moe_small_t_kernels.cu:190 switch (tokens)`。

T 解析（Q4-up/Q5-down 档）：T=48 → **adaptive=true ⇒ decode small-t 专家核**；T=128 → grouped prefill（narrow）；T=1200 → grouped prefill（wide）。→ **家族级分叉**。
`35B-A3B-NInfer/artifact-manifest.json` 的 `weights_id` 是 `groupwise-int`（非 nvfp4），所以**如果被测模型是 35B-A3B，这一条压倒 §1.2 成为第一嫌疑**。

### 1.5 其它权重档（对 nvfp4 档不生效，列出以免漏检）

| 文件:行 | 谓词/路由 |
|---|---|
| `linear_add/w8/w8_linear_add_plan.cpp:23-24`（kK4096） | `{2,48}->SplitKMmaExactT, {49,128}->MediumSplitK, {129,640}->MmaR32C128, {641,∞}->MmaR64C128` |
| `linear_add/w8/w8_linear_add_plan.cpp:29-35`（kK6144） | 同上 + 大量 `MmaR32C64/96/112/128`、`MmaR64*`、`MmaR128*` 精确 T 区间 |
| `linear_add/q5/q5_linear_add_plan.cpp:41-52` | `{1,1}->GemvResidual, {2,13}/{2,16}->Split2ExactResidual, {14,32}->MmaResidualR64C16, {33,48}->R64C24, {49,128}->R64C64, {129,∞}->R64C128` |
| `gdn_input_proj/w8/w8_gdn_input_plan.h:14,20` + `wrapper/gdn_input_proj.cpp:555,710` | `SplitKMmaDirect` / `SplitKMmaFused` |
| `linear/fp8/fp8_dispatch.cpp:42,44,46,58` | `tokens >= 12 / 11 / (==1 \|\| >=5) / 25 ? A8 : A16` |
| `attn_input_proj/fp8/fp8_attn_input_plan.cpp:23` | `tokens >= 11 ? A8 : A16` |
| `linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:25` | `tokens == 1 \|\| tokens >= 3 ? A8 : A16` |
| `linear/fp8/fp8_a16_gemm_mma.cu:91` | `if ((tokens >= 161 && tokens <= 192) \|\| (tokens >= 257 && tokens <= 288))` |
| `gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp:45,88,106` | `const bool fused = width <= 3 \|\| (width >= 7 && width <= 10);` / `width >= 10` |
| `linear_add/bf16/bf16_linear_add_plan.cpp:17` | `if (tokens == 1) { ... Decode; }` |

---

## 2. 逐点判据（代码事实）

### 2.1 nvfp4 MMA 家族：**不会分叉**（重要负面结论）

`src/ops/linear/nvfp4/nvfp4_w4a4_mma.cuh`：
- `:220 constexpr int kKTiles = Geometry::kInputRows / Schedule::kBlockK;`
- `:234 float accumulators[Schedule::kMmaM][Schedule::kMmaN][4] = {};` —— **单一累加器链，无 split-K、无 atomic、无 partial 缓冲**
- `:248 for (int k_tile = 0; k_tile < kKTiles; ++k_tile)` → `:254 for (int local_k64 = 0; local_k64 < Schedule::kK64PerStage; ++local_k64)` → `:308 mma_nvfp4_e4m3(accumulators...)`
- 六个档位（`gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu:14-18`）**BlockK 全部 = 256** ⇒ `kK64PerStage = 4`（`nvfp4_w4a4_mma.cuh:40`），全局 k64 序号 = `k_tile*4 + local_k64` 恒为 0,1,2,…，顺序与分组逐位相同。
- 网格只沿 `blockIdx.y`（token 块）与 `blockIdx.x`（输出行块）拆分（如 `nvfp4_w4a4.cu:24-25`），同一 token 行的激活（codes/scales）与权重行都不跨块 ⇒ 每个输出元素的计算序列与块划分无关。
- 越界 token 行被零填充（`:96 cp_async_zfill<16, Cache::cg>(destination, input, valid ? 16 : 0)`）且 store 处 `:373 if (token < tokens)` 拦截 ⇒ 行填充不污染。

⇒ §1.1 里 T=48/128/1200 命中不同 schedule（M32N64 vs M128N128Pipelined vs M128N128Resident）**不会**产生数值差异。
（`blocked_scales` 只改 scale 的行主序/瓦片连续布局：`nvfp4_w4a4_mma.cuh:404-429` 与 `:392-402` 写的是同一个 `quantized.scale`，值不变。）

同时证明 **`FusedW4A4`(T≤48) ≡ baseline `linear()`+`silu_mul`**：
- MMA 融合核先按 `:360-361 __floats2bfloat162_rn(value00, value01)` 把 gate/up 落成 bf16 到 shared，
  再由 `nvfp4_linear_swiglu_w4a4.cu:41-50 combine()` 用 `__bfloat1622float2` 读回 → `silu(g)*u` → bf16。
- baseline：`linear()` 落盘 bf16（同 `__floats2bfloat162_rn`），`silu_and_mul.cuh:18-22 silu_mul_pair` 读 bf16 → `silu()`（同一个 `ops/common/math.cuh:19 silu`）→ bf16。
⇒ 两者逐位相同。**所以 T=48 块本身不是分叉源；分叉源是 TMA epilogue 少了那一次 bf16 回舍。**

### 2.2 MLP gate-up：**确定会分叉**
见 §1.2。T=1200 时 tokens 0..1023 走 TMA（无回舍），T=128 时同名 token 走 baseline（有回舍）——
同一批 token 走了不同精度路径。中间 1024..1151 两边都是 baseline（一致），1152..1199 两边也都等价（T=48 MMA ≡ baseline）。
⇒ 只动 `--prefill-chunk` 就会让 **tokens 0..1023 的 MLP 激活出现 ~2^-9 级差异**，并沿因果链传到第 1199 个 token 的 hidden。

### 2.3 GDN gating proj：**确定会分叉**
见 §1.3。`Split8` vs `Split4` 的 K 切片边界不同、partial 个数不同，归约是 fp32 加法 ⇒ 同一 token 的 `g`/`beta` 不同。

### 2.4 rope：**不会分叉**
`src/ops/launcher/rope.cu:51/53/55`（`tokens <= 6` / `<= 1020`(kLargeBlockWaveCapacity) / `<= 1024`）与
`:85/87/90`（DFlash `<= 16` / `<= 400` / else）只改 **block 大小 / HeadsPerBlock / grid**：
`rope.cuh:197-203` 每块只用 `fixed_sincos<Mode>(positions, tokens, token, pair, ...)` 生成 cos/sin 缓存（同一函数、同一 token），
`rope.cuh:216 for (int combined_head = warp; combined_head < QHeads+KHeads; combined_head += block_warps)` 每个 head 独立做
`rope.cuh:167-180 apply_rope_head`。本用例 27B 文本 rope（24Q/4K, D64）：T=48/128 → block=256，T=1200 → block=128，head 循环步长变、结果不变。

### 2.5 causal_conv1d（GDN 卷积）：**不会分叉**
`launcher/causal_conv1d.cu:264/274/284`：`B==1 && T==1` → decode；`T <= kCausalConvParallelMaxTokens(32)` → smallt；否则 → sequence。
`launcher/causal_conv1d.h:20 inline constexpr std::int32_t kCausalConvParallelMaxTokens = 32;`
`wrapper/causal_conv1d_silu.cpp:268/270/273` 与 `:332/334/337` 同构。
T=48/128/1200 **全部 > 32** ⇒ 恒走 `causal_conv1d_prefill_launch` / `..._sequence_snapshot_kernel`，同一内核。

### 2.6 KV 写入（`tokens >= 128` 分档）：**不会分叉**
`kv_cache/append/launch.cu:24`（fp8）与 `:57`（i8）：`if (tokens >= 128 && Geometry::KVHeads == 2)`
→ page 变体 `kv_cache_append_full_{fp8,i8}_page_kernel`（`kernel.cuh:169/:266`）vs 单元变体（`kernel.cuh:141/:201`），
两者对每个 (token, kv_head) 调**同一个** `kv_cache_append_full_fp8_row<Geometry>`（`kernel.cuh:56-104`，page 版调用点 `:197`，单元版 `:164`），
量化在 `:65-103` 完全一致（同一 `warp_max` 组内 64 元素、同一 `kv_cache_fp8_quant_params`）。
`launcher/gqa_attention_prefill.cu:261`、`gqa_attention_prefill_e8.cu:50` 的同名分档同理：
`kernel/gqa_attention_prefill_i8.cuh:87-160`(单元) 与 `:224-300`(page) 的 `d0 = group * kGqaKvQuantGroup + lane`、`d1 = d0+32`、
`kGqaPrefillI8Groups == 4`（`:49 static_assert`）与 grid 的 `kGqaKvQuantGroups` 一致 ⇒ 逐位相同。
另外这两个 page 变体只在 `KVHeads == 2`（`gqa_attention_prefill_i8.cuh:281` 注释：Muse 几何）下才可能被选中，27B 是 24Q/4KV ⇒ **恒不生效**。

### 2.7 注意力路由与小 T 家族：**不会分叉**（prefill 段）
`wrapper/gqa_attention.cpp:19-23`（`kSmallTChunkTokens = 6`, `kMaximumVerifyTokens = 16`,
`kTwoChunkPromptVisibleKeys = 512`, `kThreeChunkPromptVisibleKeys = 1024`）、`:404-412`：
`width 1..6 -> SmallT`；`batch_size > 1 -> ChunkedSmallT`；`q_heads == 16 && width <= 16 && envelope.max_visible_keys > prompt_visible_keys -> ChunkedSmallT`；否则 `Prompt`。
本用例 batch=1、width∈{48,128,1200}、27B 的 q_heads=24（`causal_softmax_attention.cpp:28-29` 只允许 24/4 与 16/2）⇒ **两边都 Prompt**。
Prompt 的 grid 是 `prompt.cu:49 div_up(tokens, kCausalPromptBr) × Geometry::QHeads`，**没有 key 轴拆分维**；
split 只在 SmallT/ChunkedSmallT 分支出现（`gqa_attention.cpp:352-395` 的 `for (begin ...; begin += kSmallTChunkTokens)`）。
同族且对 prefill 不生效的 T≤16 域：`softmax_attention/dense/context/launch.cu:26-41`（`case 1..16`）、
`context_softmax_attention.cpp:52 if (tokens < 1 || tokens > 16) throw`、`sliding_window/launch.cu:21`、
`launcher/swa.cu:21`、`launcher/bidirectional_gqa_attention.cu:63 (tokens <= 8)`、
`causal_cache/small_t.cu:236 bool causal_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }`。
（这些只影响 decode/verify 的 T≤16 步，不影响第 0 个 token。）

### 2.8 lm_head：**不会分叉**（对第 0 个 token）
`targets/qwen3_6/impl/runtime/text_context_impl.h:1405-1408`：
```
if (is_last) { Tensor last_xf = xf.slice(1, len - 1, 1); Tensor logits = matrix_window(io_.logits, 1);
               ops::linear(last_xf, *lm_head_, logits, s); }
```
⇒ 两次运行中 lm_head 都是 **T=1** 的 GEMV。第 0 个 token 的差异只能来自喂进去的 hidden，不可能来自 lm_head 的 T 分档。
（唯一会以 T>1 跑 lm_head 的是诊断路径 `text_prefill_impl.h:155 ops::linear(normed, output_head, logits)`，
只在 `NINFER_HS_DUMP_TOPK` 且 `tokens <= 1024` 时触发，即 1200 块**不产生** top-k 段而 128 块产生——这是 dump 的不对称，不是引擎差异。）

### 2.9 rmsnorm / l2norm：**不会分叉**
`launcher/rmsnorm.cu:23,34,43,51,58,66` 与 `launcher/l2norm.cu:26,35` 的分派谓词只含 `d`（=ne[0]=hidden）与 `aligned2`，
不含 rows/tokens；`blocks = div_up(rows, ...)` 只做行切分 ⇒ 每行归约顺序不变。
其它逐元素算子（`residual_add`/`silu_mul`/`cast`/`add_bias`/`embedding`/`scatter`/`argmax`/`sampling`）无 row-wise 归约，只会换内核形状。

### 2.10 GDN chunked vs recurrent tail：**本用例不变**
`linear_attention/gated_delta_net/gated_delta_net.cpp:248-289`：
`T_full = (T / kChunkSize(64)) * kChunkSize; tail = T - T_full;`
`if (T_full > 0) launch_chunked(...)`；`if (tail > 0) launch_recurrent_inout(..., tail_in = (T_full>0) ? ssm_state_out : ssm_state_in, ...)`。
T=128 → T_full=128/tail=0；T=48 → T_full=0/tail=48（读 ssm_state_in）；T=1200 → T_full=1152/tail=48（读 ssm_state_out）。
**两种情况末 48 个 token 都走 recurrent 路径、入口 state 都是上一个 chunk 发布的 BF16 `ssm_state`** ⇒ 与主代理已修的
"发布 BF16 vs 更宽累加器" 不是新增变量。（`nvfp4_gdn_snapshot_plan.cpp:38/42/43` 的 `tokens==1 / <=16 / else Materialized`
在 T≥48 时恒 `Materialized`，与已修阈值一致。）

---

## 3. 排序表 + 最小对齐方案 / 性能代价

| # | 源 | 判据 | 最小对齐方案 | 性能代价 | 单变量判别实验 |
|---|---|---|---|---|---|
| 1a | MoE `adaptive` (`sparse_moe_prefill_kernels.cu:1153`) | 家族级不同（decode small-t vs grouped prefill） | 把 `adaptive` 窗口改成按绝对位置（如只对 decode 调用生效），或让 prefill 的 `tokens<=adaptive_last` 段也走 grouped 路径；最省事是在 `sparse_moe_uses_prefill` 层面拒绝 T<`kSparseMoePrefillQ4Q5Min` 的 prefill 调用 | 低（只影响 T≈47..52 的 few-token 尾块，可能慢一点） | `--prefill-chunk 128` vs `--prefill-chunk 46`（46<47 → 整段走 decode 路径）若第 0 token 一致，说明 adaptive 不是源 |
| 1b | MLP gate-up TMA epilogue 缺 bf16 回舍（`nvfp4_linear_swiglu_w4a4_tma.cuh:242-245`） | 值级：同一 token 有/无一次 bf16 回舍 | 在 `:242-245` 令 `gate/up` 先 `__bfloat162float(__float2bfloat16(g*alpha))` 再 silu；或反过来在 MMA/baseline 端保留 fp32（成本更高，需 fp32 projected 缓冲） | **≈0**（每 8 个输出多 8 条 CVT） | `--prefill-chunk 128`(永不 TMA) vs `--prefill-chunk 512`(恒 TMA；512<1024 ⇒ linear/attn 仍非 TMA，GDN gating 仍 Split8) —— **最干净的单变量**；prompt 取 1024 两边整除 |
| 2 | GDN gating SplitK 随 T（`bf16_gdn_gating_proj_plan.cpp:35-38`） | fp32 归约分组不同 | 固定 `split_k`（对所有 T 用 8），或把 partial 归约改成与 split 数无关的规范顺序 | 中：`MmaCooperativeSplit4/2` 是为了 cooperative grid 可驻留（`:32-34` 注释），强改 Split8 可能触发驻留/launch 约束，需要重测 occupancy | prompt 1200：`--prefill-chunk 4096`(Split4) vs `--prefill-chunk 1023`(Split8+Split8，1023 非 256 倍数 ⇒ 无 TMA) |
| 3 | MoE `wide_plan`(768) → `route_job_bn` 32/64（`:1165-1166`） | 分组瓦片粒度变，无 K 拆分 | 固定 `route_job_bn = 32` | 低（只在 T≥768 时慢一点） | prompt 1200: `--prefill-chunk 4096` vs `--prefill-chunk 767` |
| 4 | w8/q5/fp8 路由表（§1.5） | 含 `SplitKMma*` 等 K 拆分档 | 仅当权重档切换时才有意义；nvfp4 artifact 不需要动 | —— | 检查 artifact `weights_id`：`nvfp4` → 免疫 |

---

## 4. `NINFER_HS_DUMP_DIR` 覆盖范围与命名规则

代码：`src/targets/qwen3_6/impl/runtime/text_prefill_impl.h:85-204`（`dump_prefill_chunk`）、触发点 `:300-305`。

**触发条件（两个都必须满足）**
1. `NINFER_HS_DUMP_DIR` 非空（`:92-93` 缓存到 static）。
2. `state.dflash != nullptr || state.dflash2 != nullptr`（`:300-301`）——**只有 DFlash/DFlash2 编排的 prefill 才会 dump**；纯 target prefill 不产生任何文件。
   调用位置：`prefill_text_chunk` 末尾，**每个 prefill chunk 一次**（`:302-304`，`tokens = result.processed_tokens`）。

**命名**
```
path = <dir>/chunk_<7 位十进制>.bin     // :122-126，逐位拼 id/1000000%10 … id%10
++counter 是函数内 static std::atomic<std::uint32_t>   // :94,:121 —— 进程内的 dump 序号，与 chunk 序号一致
```
⇒ `--prefill-chunk 128`（M=1200）产出 **10 个文件**：`chunk_0000000..0000008`（各 128 token）+ `chunk_0000009`（48 token）；
`--prefill-chunk 4096` 只产出 **1 个文件** `chunk_0000000`（1200 token）。

**文件内容（写序）**
```
u32 magic = 0x4E485331 ("NHS1")                      // :129-130
i32 tokens                                           // :131（该 chunk 的 token 数）
i32 ids[tokens]   = 该 chunk 的 prompt token id（从 ids[chunk_base + i]） // :110-113,:132
u16 feat[feature_rows x tokens]   BF16，来自 dflash->prefill_features / dflash2->prefill_features  // :102-106,:133
u16 last[hidden x tokens]         BF16，来自 execution.prefill_hidden = **post-final-norm hidden** // :107-109,:134
```
可选尾巴（**仅当** `NINFER_HS_DUMP_TOPK` 非空 **且** `tokens <= 1024`，`:140`）：
`i32 top1[tokens]`、`i32 marker(=16)`、`i32 ids16[tokens*16]`、`u16 vals16[tokens*16]`（`:197-201`），
其 logits 由 `rmsnorm(xwin, final_norm) + linear(normed, output_head)` 现算（`:153-155`，临时 `cudaMalloc`）。

`feat` 的 5 层来源（`src/targets/qwen3_6_27b/impl/config.h:95-96,113` / `:126-127,150`）：
`feature_layers = 5`，`feature_rows = 5 * hidden`，
DFlash `target_feature_layers{4, 16, 28, 40, 52}`，DFlash2 `{5, 19, 33, 47, 61}` —— **按 capture 顺序拼接，无偏移表头**。

**主代理能看到什么**
- 每个 chunk 的**全部 token** 的 post-final-norm hidden（`last`）→ 按绝对位置对齐后（chunk 128 的第 k 块对应单块的 `[128k, 128k+128)` 列），可直接定位**第一个出现差异的 token 列**，也就是整条 60 层栈之后的分叉落点。
- 5 个被 capture 的层（4/16/28/40/52 或 5/19/33/47/61）的 hidden（`feat` 的对应切片）→ 能把分叉夹到"哪一段层区间"。
- （配合 `NINFER_HS_DUMP_TOPK`）引擎自己的 logits top-16 → 判断第 0 个生成 token 的 argmax 是否由 hidden 差异引起。

**看不到什么**
- **任意层的 hidden 都没有**；`feat` 只有上面 5 层，且**没有层号/偏移字段**，必须自己按 `5*hidden` 切。
- **没有任何 GDN 内部量**：`a/g/beta`（gating proj 输出）、conv 输入/输出、conv_state、`ssm_state`、chunked 路径的 `h_chunk`、`q/k/v/z`、`attn_input_proj`/`o_proj`/`gate_up`/`down` 的中间 buffer 全都不 dump。
- 没有 attention 输出、没有 rope 前后、没有 per-layer residual。
- 没有 KV cache（那是另一条路径 `NINFER_KVDUMP_DIR`，`:206-276`：`kvc_<id>_t<tokens>_bt.bin` + `_L<layer>_{k,v,ks,vs,kr,vr}.bin` + 各自 `_meta.txt`，层默认 `15,16`、页数默认 4）。
- `NINFER_HS_DUMP_TOPK` 段**只有 tokens<=1024 的 chunk 才有** ⇒ 128 分块下每块都有 top-k，单块 1200 下**没有** top-k 段（解析时按 `tokens` 判断）。
- dump 为每个 chunk 做 `cudaStreamSynchronize`（`:120`），只影响时延不影响数值。
- 缓冲不一致时 **抛异常**（`:98-101`），不会静默出半截文件。

---

## 5. 附：本次未验证完的相邻风险（供主代理取舍）

- `nvfp4_w4a4_tma.cuh` 的 `Nvfp4IdentityEpilogue` / `Nvfp4AddResidualEpilogue` 是否与 MMA 家族逐位一致（本次只验证了 **swiglu** 的 TMA epilogue 少回舍）。本用例 T=1200 不触发（`1200 % 256 = 176`），但 **T=1024 / 1280 / 1536 / 2048 会触发**，且 chunk=256·k 的运行都会命中。
- MoE `wide_plan` 与 grouped-route 的专家累加顺序未逐行核对（`route_job_bn` 只改分组瓦片，理论上无 K 拆分）。
- `linear_add/w8`、`q5`、`fp8` 路由表的 K 拆分档对非 nvfp4 artifact 是**确定会分叉**，需要按 artifact `weights_id` 决定是否排查。
