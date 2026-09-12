# PART-A 报告：split-K 的 `partial_acc` BF16 → FP32（统一算术·精度档对齐）

日期：2026-09-11 · 树：`/home/user/ninfer-fusion`（**未改**：只读 + scratch 副本 + `--dry-run`）
产物：`_collab/PART_A_patch.diff`（21 文件 / 61 hunk / 755 行）、本报告。
验证：`cd /home/user/ninfer-fusion && patch -p1 --dry-run < PART_A_patch.diff` → **21/21 checking，exit 0**；
无 `.rej` 产生，构建树 mtime 未变。

---

## 0. 结论先行

| # | 结论 | 证据 |
|---|---|---|
| 1 | **R31 的定位成立**：所有 split-KV 归约的 `partial_acc` 都是 BF16，`partial_m/l` 是 FP32。已把 **全部 5 个 KV 档**（bf16 / i8 / E8 / nvfp4 / fp8 / iso3）×（partial 核 + reduce 核 + 分配）统一升 FP32。 | 5 个分配点 `workspace.alloc(DType::BF16, {head_dim, q_heads, tokens, splits*batch})` → `FP32` |
| 2 | **预判之外的关键事实**：27B 的 **KV 默认档不是 BF16**，是 `NVFP4`（E2M1 K + ISO3 V）+ 10 层 `E8Kv`。所以热的 partial 核是 `gqa_attention_decode_nvfp4.cuh` 与 `gqa_attention_decode_i8.cuh<E8=true>`，**`gqa_attention_decode_bf16.cuh` 只在 `--kv-dtype bf16` 时才是活路径**。本 patch 两条都覆盖（否则很可能测不到任何变化）。 | `targets/qwen3_6_27b/impl/variant.cpp:23-39`；`launcher/gqa_attention_decode_e8.cu:37,43`（E8 → i8 核）|
| 3 | **verify(T=8) 侧根本没有这个缓冲**：width=8 > `kSmallTChunkTokens=6` ⇒ route=Prompt ⇒ `gqa_attention_workspace_capacity_bytes` 该分支返回 **0 字节**。⇒ **本改动对 verify 的吞吐/占用是 0 影响**（不满足"禁止 verify 退化"的可测空间）。 | `wrapper/gqa_attention.cpp:404-412, 473-477` |
| 4 | 代价只在 **T≤6 的 decode/MTP** 与 **DFlash/DFlash2 draft 注意力**：单实例缓冲 ×2，最坏 +1 MiB；per-token 流量 +≈25 MiB（ctx 4096，17 次注意力）≈ **0.16%**（111 tok/s 基准）。**无 smem/寄存器/occupancy 恶化，反而少一次 `__syncthreads()`**。 | §3 |
| 5 | 唯一让步：bf16/fp8/iso3 三个核原本用 **128-bit `int4` 打包 store**（靠 BF16 shared 中转），升 FP32 后改为 **直接 `float2`（8 B）**——保留 128-bit 需要 +16 KB 静态 smem 的 FP32 中转，反而扰动 occupancy。判据与替代方案见 §4。 | §4.1 |
| 6 | 同族未改：`ops/softmax_attention/dense/causal_cache/`（bf16+i8）。**全树无调用者**（`ops::causal_softmax_attention*` 只在自己的 .cpp/.cu 内出现）；且其 fp8 档 `small_t_fp8.cuh:580` **早就是 FP32 + `make_float2`** —— 本 patch 正是把其余档对齐到这个既有正确实现。改法清单见 §4.2。 | grep 证据见 §4.2 |

---

## 1. 改动面：逐点表格（行号为 **改前** 行号）

### 1.1 类型 / 指针签名（部分 = 写侧）

| 文件:行 | 改前 | 改后 |
|---|---|---|
| `ops/softmax_attention/common/context_query.cuh:95` | `__nv_bfloat16* __restrict__ partial_acc` | `float* __restrict__ partial_acc` |
| `ops/softmax_attention/dense/context/kernel.cuh:77` | `... float scale, __nv_bfloat16* __restrict__ partial_acc,` | `... float* __restrict__ partial_acc,` |
| `ops/softmax_attention/sliding_window/kernel.cuh:47` | `__nv_bfloat16* __restrict__ partial_acc` | `float* __restrict__ partial_acc` |
| `ops/kernel/bidirectional_gqa_attention.cuh:116,528` | `... float scale, __nv_bfloat16* __restrict__ partial_acc,` | `... float* __restrict__ partial_acc,` |
| `ops/kernel/bidirectional_gqa_attention.cuh:543`（`swa_split_partial_kernel`） | `__nv_bfloat16* __restrict__ partial_acc` | `float* __restrict__ partial_acc` |
| `ops/kernel/gqa_attention_decode_bf16.cuh:25` | `__nv_bfloat16* partial_acc, ...` | `float* partial_acc, ...` |
| `ops/kernel/gqa_attention_decode_i8.cuh:81` | 同上 | 同上（E8 复用此核） |
| `ops/kernel/gqa_attention_decode_nvfp4.cuh:138` | 同上 | 同上 |
| `ops/kernel/gqa_attention_decode_fp8.cuh:31` | 同上 | 同上 |
| `ops/kernel/gqa_attention_decode_iso3.cuh:32` | 同上 | 同上 |

### 1.2 打包存储点（宽度变化）

| 文件:行 | 改前（宽度） | 改后（宽度） |
|---|---|---|
| `common/context_query.cuh:449,461` | `store_vec(&partial_acc[dst], pack_bf16x2(a,b))` = **4 B** | `store_vec(&partial_acc[dst], make_float2(a,b))` = **8 B** |
| `kernel/bidirectional_gqa_attention.cuh:503,515`（`store_vec`/`pack_bf16x2`） | **4 B** | **8 B** `make_float2` |
| `kernel/gqa_attention_decode_bf16.cuh:399-426` | BF16 shared 中转 → `store_vec(int4)` = **16 B/8 元素**（+2 条 smem store/元素 +1 `__syncthreads`） | 直接 `store_vec(float2)` = **8 B/2 元素**，删中转、删 `__syncthreads` |
| `kernel/gqa_attention_decode_fp8.cuh:482-509` | 同上（int4 16 B） | 同上（float2 8 B，删中转） |
| `kernel/gqa_attention_decode_iso3.cuh:539-566` | 同上（int4 16 B） | 同上（float2 8 B，删中转） |
| `kernel/gqa_attention_decode_i8.cuh:802,810` | `*reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(...)` = **4 B** | `*reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(...)` = **8 B** |
| `kernel/gqa_attention_decode_nvfp4.cuh:1010,1018` | 同上 = **4 B** | 同上 = **8 B** |

（`float2` 对齐前提，可静态证明：`dst = d0 + HeadDim*(q_head + QHeads*(token + T*split))`，`d0 = 2*lid + 8n` 恒为偶、`HeadDim ∈ {128,256}` 恒为偶 ⇒ `dst` 偶 ⇒ `4*dst` 是 8 的倍数 ⇒ `float2` 自然对齐。）

### 1.3 中性初始化（零）

| 文件:行 | 改前 | 改后 |
|---|---|---|
| `decode_bf16.cuh:102-103`、`decode_fp8.cuh:108-109`、`decode_iso3.cuh:109-110`、`decode_i8.cuh:183-184` | `= __float2bfloat16(0.0f);` | `= 0.0f;` |
| `decode_nvfp4.cuh:250-252`（宏 `NINFER_NVFP4_WRITE_NEUTRAL`） | 同上（注意保留行尾 `\` 续行） | 同上 |

### 1.4 reduce（读侧）

| 文件:行 | 改前 | 改后 |
|---|---|---|
| `common/context_query.cuh:469` | `context_query_reduce_body(const __nv_bfloat16* ...)` | `const float*` |
| `common/context_query.cuh:528-531` | `numerator += __bfloat162float(partial_acc[...]) * weight;` | `numerator += partial_acc[...] * weight;`（去转换，累加本就是 FP32） |
| `dense/context/kernel.cuh:93` | `context_attention_reduce_kernel(const __nv_bfloat16* ...)` | `const float*` |
| `sliding_window/kernel.cuh:67, 130-133` | `const __nv_bfloat16*` / `__bfloat162float(...)` | `const float*` / 去转换 |
| `bidirectional_gqa_attention.cuh:553, 621-623`（`noncausal_gqa_reduce_body`） | `const __nv_bfloat16*` / `__bfloat162float` | `const float*` / 去转换 |
| `bidirectional_gqa_attention.cuh:632`（`bidirectional_gqa_reduce_kernel`） | `const __nv_bfloat16*` | `const float*` |
| `bidirectional_gqa_attention.cuh:646, 709-712`（`swa_reduce_kernel`，被 `ops::swa` 用） | 同上 | 同上 |
| `gqa_attention_decode.cuh:149, 247-250`（**6 个 KV 档共用**的 reduce） | `const __nv_bfloat16* partial_acc` / `__bfloat162float(...)` | `const float*` / 去转换 |

### 1.5 分配点（workspace dtype，唯一决定字节数的地方）

| 文件:行 | 形状（元素数） | 改前 | 改后 |
|---|---|---|---|
| `wrapper/gqa_attention.cpp:342` | `head_dim×q_heads×tokens×splits*batch` | `DType::BF16` | `DType::FP32` |
| `wrapper/bidirectional_gqa_attention.cpp:83` | `kHeadDim×kQHeads×tokens×splits*batch` | `DType::BF16` | `DType::FP32` |
| `wrapper/swa.cpp:68` | 同上 | `DType::BF16` | `DType::FP32` |
| `softmax_attention/dense/context/context_softmax_attention.cpp:90` | 同上 | `DType::BF16` | `DType::FP32` |
| `softmax_attention/sliding_window/sliding_window_attention.cpp:78` | 同上 | `DType::BF16` | `DType::FP32` |

**索引/步长未动**：`gqa_partial_acc_index`、`context_query_partial_index`、`bidirectional_gqa_partial_index` 都是"元素下标"，`partial_acc += QueryElements*split_capacity*batch` 也是元素数，dtype 变化后**无需**改。`partial_m/l`（FP32）**未动**。路由/分派**未动**（`gqa_attention_resolve_route`、`context_attention_resolve_plan` 等零改动；容量函数因复用同一 `allocate_workspace` 自动跟着变，无需手改任何字节常量——已全树确认无手工 partial 字节核算）。

**覆盖核对（机械校验，非人眼）**：把 `src/CMakeLists.txt` 的 `ninfer_ops` 源列表中所有提到 `partial_acc` 的 TU、以及全树提到 `partial_acc` 的头文件，逐个查是否在 diff 里：

| 类别 | 结果 |
|---|---|
| 被编译且提到 `partial_acc` 的 TU 共 8 个 | 6 个已改（decode / decode_e8 / bidirectional / swa / context·launch / sliding_window·launch）；2 个未改属 `causal_cache`（§4.2，自成闭环且不可达） |
| 提到 `partial_acc` 的头文件共 17 个 | 10 个"写/读该缓冲"的头全部已改；7 个未改全部是**签名透明**的：`{context,sliding_window}/launch.h`、`launcher/{gqa_attention,swa,bidirectional_gqa_attention}.h` 只传 `Tensor&`（dtype 无关，**本就不需改**），`launcher/gqa_attention_decode_{partial,impl}.cuh` 是 §4.3 的幽灵副本，`causal_cache/*` 见 §4.2 |

### 1.6 发射点类型转换（`partial_acc.data` only）

`dense/context/launch.cu:128,156,162` · `sliding_window/launch.cu:118,137,147` · `launcher/bidirectional_gqa_attention.cu:128,156,162` · `launcher/swa.cu:116,135,145` · `launcher/gqa_attention_decode.cu:157,190,225,280,414,627` · `launcher/gqa_attention_decode_e8.cu:60`
—— 共 19 处 `static_cast<__nv_bfloat16*>(partial_acc.data)` / `static_cast<const __nv_bfloat16*>(...)` → `float*` / `const float*`（同文件里 q/k/v/out/cache 的 bf16 转换**未动**）。

---

## 2. 受影响张量：元素数与字节数（改前 / 改后）

27B 几何：`HeadDim=256, QHeads=24, KVHeads=4, GroupSize=6, DecodeSplits=85, full_attention_layers=16, mtp_layers=1`。
split 数按源码头重放（`gqa_small_t_split_count` + `gqa_small_t_launch_capacity` + `context/bidirectional/swa_resolve_plan`）：
BF16/FP8/ISO3 走 `gqa_small_t_split_upper_bound`（段端点 4096/8198/16390）；NVFP4 走粗档 target(64/256/480)；I8/E8Kv 的 T=5/6 特例不适用于 T=1。下表 split 取 **envelope 区间**在该 ctx 处的值：

| 路径 | ctx | dtype | splits | 元素数 | BF16 | FP32 | Δ |
|---|---|---|---|---|---|---|---|
| plain decode T=1 | 1024 | BF16/NVFP4/E8 | 16 | 98,304 | 192 KiB | 384 KiB | +192 KiB |
| plain decode T=1 | 4096 | BF16/NVFP4/E8 | 64 | 393,216 | 768 KiB | 1.50 MiB | **+768 KiB** |
| plain decode T=1 | 16384 | BF16(NVFP4=64) | 65(64) | ~393k | ~768 KiB | ~1.5 MiB | ~+768 KiB |
| plain decode T=1 | ≥131072 | 全档 | 85 | 522,240 | 1020 KiB | 1.99 MiB | +1020 KiB |
| MTP/短 T (T=2) | 4096 | 全档 | 64 | 786,432 | 1.5 MiB | 3.0 MiB | +1.5 MiB |
| MTP/短 T (T=6) | 4096 | 全档 | 64 | 2,359,296 | 4.5 MiB | 9.0 MiB | +4.5 MiB |
| **verify T=8** | 任意 | — | — | **0** | **0** | **0** | **0** |
| `context`/`sliding_window`/`bidirectional`/`swa`（128×32） | 任意（分裂上限 32） | — | 32 | 131,072 | 256 KiB | 512 KiB | +256 KiB |

注：`splits` 用的是 **envelope 区间最大值**（`gqa_small_t_launch_capacity` 扫段端点），所以图内区间越宽、分配越大；上表按"区间端点=该 ctx"取值。

---

## 3. 性能代价（逐项判断）

**(a) 字节数**：上表，单实例 +192 KiB … +1.0 MiB（T≤6 通路），verify 侧 +0。

**(b) 向量化 store 宽度**：
- `int4`(16 B)→`float2`(8 B)：store 指令数 ×2、字节 ×2 —— 但那只在**每次 launch 一次**的收尾写出，元素总量 ≤ 2.4 M（T=6），占总流量可忽略。
- `unsigned`(4 B, 2×bf16)→`float2`(8 B)：指令数不变、字节 ×2。
- 代价的"真身"是 **partial 缓冲的写 + 读各 ×2**。

**(c) 每 token 的额外流量**（27B，plain T=1，ctx 4096，`full_attention_layers=16` + MTP 1 = 17 次注意力）：
`17 × 2 × 768 KiB ≈ 25.5 MiB/token`；5090 ≈1.79 TB/s ⇒ **≈15 µs**；基准 111 tok/s ⇒ token 时间 ≈9.0 ms ⇒ **≈0.17%**（R31 预估 <1% 成立）。DFlash2 draft（`swa`，Δ256 KiB/实例）再加 ≤ 数 MiB/token。
> 口径限定：R30 的 setup 是 verify K=7 ⇒ T=8 ⇒ Prompt。**若把 draft 数调到 K≤5（T≤6），verify 也会走 SmallT 并分配该缓冲**，那时代价按上表 T=2/T=6 行计（+1.5~4.5 MiB/实例）——但仍只在 T≤6 通路，且此时 plain 与 verify 两边用同一精度档，正是本改动要的效果。

**(d) register / shared / occupancy 判断**：
- **无 smem 变化**：`qkv_s`/`p_s`/`dynamic_r_s` 声明与尺寸全未动；bf16/fp8/iso3 反而**少用**了原先借 `qkv_s` 做 FP32 中转的那些 store（并**删掉一次 `__syncthreads()`**）。
- **寄存器**：净变化 ≤ ±1（`d1` 变量被删；`float2` 值占 2 个寄存器 vs `unsigned` 1 个；同时少掉 `__bfloat162float` 与 smem load）。
- **occupancy**：`__launch_bounds__(128,2)` / `(256)` 与动态 smem 请求**均未改** ⇒ **编译后占用不变**（须实测确认，见 §6）。
- **不允许发生的事**：verify 不退化 ✓（它不分配此缓冲）；大缓冲未升 FP32 ✓（只动 partial）；路由未动 ✓。

---

## 4. 无法"零代价"升 FP32 的点与替代方案

### 4.1 唯一让步：128-bit 打包 store 在 bf16/fp8/iso3 上从 `int4` 降为 `float2`

这三个核原先的写法是：MMA 寄存器 → **BF16** 写 shared（`qkv_s`）→ `__syncthreads()` → 跨线程 gather → `int4` 一次写 8 个 BF16。
升 FP32 后：
- **保 128-bit 的方案**：把中转缓冲改成 FP32（`QkvRows*D` floats = 64×128×4 = **32 KiB**，而 `qkv_s` 现在是 16 KiB 且前半还要当 K/V tile 用 ⇒ 必须**净增 ~16 KiB 静态 smem**），再 `float4`(16 B)写 4 个元素。
  **判据**：只有当 `ptxas --resource-usage` 显示该核 smem 增量后 `registers/spill` 不变、且实测 decode tok/s 不降（该核 `__launch_bounds__(128,2)` 且 sm_120 单块 smem 预算紧）才值得；否则得不偿失。
- **本 patch 选用的方案**：删中转、直接 `float2`（8 B）。理由：MMA 片段布局决定了**单线程在 d 上连续只有 2 个元素**（`d0 = 2*lid`），`float2` 已是寄存器直出可达的最大宽度；128-bit 必须借 shared 中转，而中转的 BF16 往返**正是要消除的精度损失**。代价只是 store 数 ×2、字节 ×2，落在这 ≤2.4 M 元素的小张量上。
- 结论：**这三处"升 FP32 与保 128-bit"不可兼得**，已按性能优先选了 float2；若主代理实测该核变慢，按上面的判据改回 float4+FP32 中转即可（改动局部、可逆）。

### 4.2 同族未改：`ops/softmax_attention/dense/causal_cache/`（bf16 + i8）

- 未改理由：**不可达 + 自成闭环**。`ops::causal_softmax_attention` / `..._cached` 在全树只在自己目录内出现（`causal_softmax_attention.cpp:406,448`），任何 `targets/` 路径都不调用它。注意这一族**有两个 TU 确实在编译列表里**（`dense/causal_cache/small_t.cu`、`small_t_fp8.cu`）——但它们与 `small_t.cuh`/`small_t_bf16.cuh`/`small_t_i8.cuh` **不与本 patch 共享任何代码**（自带 reduce、自带索引、自带分配），保持全 BF16 或全 FP32 都是自洽的 ⇒ 不引入编译错误，也不影响活路径。（另：`small_t_fp8` 这一路本来就是 FP32，见下。）
- 参考实现已存在：`causal_cache/small_t_fp8.cuh:52`（`float* partial_acc`）、`:580`（`*reinterpret_cast<float2*>(&partial_acc[dst]) = make_float2(...)`）、`:595`（`const float*` reduce）、`small_t_fp8.cu:62`（`static_cast<float*>(partial_acc.data)`）—— 与本次改动**逐字同构**，说明 FP32 partial 是这个仓库的既定设计，BF16 档属滞后。
- 若要一并统一，需改 5 处：`small_t_bf16.cuh:27,104,507`、`small_t_i8.cuh:67,168,736,744`、`small_t.cuh:158,257`（reduce/索引读）、`small_t.cu:120,166,330`（转换）、`causal_softmax_attention.cpp` 的 `allocate_workspace` dtype。**本 patch 未含**，避免把不可达代码卷进唯一一次编译窗口。

### 4.3 必须警告的"幽灵副本"

以下文件含**同名同构的旧拷贝**，但**不在 `src/CMakeLists.txt` 的编译列表里**（已核对 `smallt|_g35|_muse|decode_partial|decode_impl` 均无命中，只有 `gqa_attention_decode.cu`、`_e8.cu`、`bidirectional_gqa_attention.cu`、`swa.cu` 被编译）：
`launcher/gqa_attention_decode_smallt.cu` · `_partial.cuh` · `_impl.cuh` · `_g35.cu` · `_muse.cu` · `*.orig`（含 `wrapper/gqa_attention.cpp.orig`）。
**风险**：若"暂存 TU 拆分"那一步把 `_smallt.cu`/`_g35.cu`/`_muse.cu` 接回构建，它们仍会写 **BF16** partial 而 reduce 已是 `float` ⇒ **静默错读（或链接到未修补的旧实例）**。⇒ 在接回前必须把这 4 个目录/文件按同一规则同步（本 patch 的规则可直接照搬：签名 + `unsigned/pack_bf16x2` → `float2/make_float2` + `__float2bfloat16(0.0f)` → `0.0f` + reduce `__bfloat162float` 去掉 + alloc dtype）。

---

## 5. 残余风险

| # | 风险 | 判据 / 缓解 |
|---|---|---|
| R1 | **只消除"BF16 分块量化"这一项**。split 与非 split 之间的 online-softmax `m/l` 合并顺序、以及 `p_frag` 在 PV-MMA 前落 BF16，**仍然存在**。所以 plain × verify 不会逐位相等，只能收敛到"FP32 结合序差"。 | 若列 0 一致率升到 90%+ 但仍非 100%，剩余差异**不应**再归因 partial；下一步按 R31 §四 转向 Prompt 侧 `p_frag`/post-attention。 |
| R2 | **不保证 solve 31% 偏移**：若 Δ 的主因是 `p_frag` 的 BF16 或 Prompt 的块内归约顺序，本改动只会部分改善。 | 把 §6-T3 的一致率当**唯一**判据，不达 90% 即按 R1 转向。 |
| R3 | **对齐性**：`float2` 依赖 `dst` 偶（HeadDim 偶、`d0` 偶）。当前 3 个几何（128 / 256）与全部 `TokenTile/GroupSize` 组合都成立；若将来引入奇 HeadDim，会 UB。 | 已在报告记录不变量；建议在 `gqa_partial_acc_index` 旁加 `static_assert(Geometry::HeadDim % 2 == 0)`（本 patch 未加，避免超出范围）。 |
| R4 | **未初始化读**：`split ≥ active_splits` 的 CTA 早退不写 partial，而 reduce 只读 `[0, active_splits)`（5 处 reduce 全满足）⇒ 平安。但这是**既有的隐式契约**：FP32 下若某处误读，`0 * NaN = NaN` 比 BF16 更容易暴露。 | 已逐处核对 5 个 reduce 的上界都是 `active_splits`；patch 后若出现 NaN，优先查这里。 |
| R5 | **workspace 峰值上升**：arena 由这 5 个容量函数推导，会自动多留 2×。CLI 会打印 `gpu workspace peak`。 | §6-T2 用它当"改动已生效"的证据；同时确认没有触发显存上限（`--gpu-mem` / `ArenaMemorySummary.capacity`）。 |
| R6 | `ft_stats`（`ft::enabled()`）读的是 `partial_l`（FP32，未动）⇒ 不受影响；但它每层做一次 D2H，性能跑必须关掉。 | `ops/launcher/gqa_attention_decode.cu:449`。 |
| R7 | 未编译验证（本轮硬约束：禁止编译）。C++ 类型层面已逐处核对：**18 处签名 + 19 处 `partial_acc.data` 转换 + 14 处 store 表达式（其中 4 处 `reinterpret_cast<float2*>`）+ 5 处中性初始化 + 5 处 alloc**（diff 统计：21 文件 / 61 hunk / +124 / −125），但仍以首个编译窗口为准。 | §6-T1。 |

---

## 6. 需主代理实测清单（按顺序）

**T1. 编译**：打 patch → 首次编译窗口。关注（i）无新的类型转换错误；（ii）**无连带的"幽灵副本"报错**（见 §4.3）；（iii）建议抓一次 `ptxas --resource-usage` 对比前后 3 个 decode 核的 `registers / smem / spill`（预期：smem 不变或略降、regs ±1）。

**T2. 改动已生效（最便宜的地面信号）**：CLI 的 `gpu workspace peak`（`apps/cli/main.cpp:211`）应比改前**上升 ≈2× 上表的 Δ**（T≤6 相位 +0.75~1.0 MiB/实例；DFlash2 proposal 相位 +256 KiB）。**若峰值完全不动 ⇒ patch 没走到活路径（或漏了分配点）**，先查这个再谈数值。

**T3. R31 §三 判据（核心）**：同 prompt，`verify 列 0 与 plain 一致率` **69% → ≥90%**（样本 ≥29）；`zh plain vs dflash2` 首次偏离后移或消失；**接受率与 decode tok/s 不得退化**（tok/s 是硬底线）。

**T4. 绑定"哪条档"的判别实验（本报告新增，最必要的一项）**：
- 同一 prompt 跑 **默认层表**（NVFP4 + E8Kv ⇒ 走 nvfp4/i8-E8 partial 核）与 **`--kv-dtype bf16`**（走 `decode_bf16.cuh`）两臂，各测 T3。
- 预期：本 patch 两臂都应改善。**若只有 `bf16` 臂改善 ⇒ 默认档的 nvfp4/E8 核还有别的 BF16 损失点**（例如 nvfp4 的 PV/P 侧、iso3 V 解码）**，应定点查那里**；若只有默认臂改善 ⇒ `decode_bf16.cuh` 那条老路径已自然被覆盖。

**T5. 噪声底**：同一二进制连跑两遍（应逐位一致）以界定一致率的观测抖动，再与 patch 前比较（R29 的 29 样本口径太窄，建议 ≥60 干净列）。

**T6. 仪器纪律**：dump 类仪器必须 `--no-cuda-graph`（`kv_calibration.h:46` 已注明，R30 也踩过）；产物写 `dl/` 或 `_collab/`（`/tmp` 随 WSL 重启清空）；收工用 `pkill -x make|nvcc|ptxas`（`-f` 会自伤）。

**T7. DFlash/DFlash2 侧**：draft 注意力走 `ops::swa`（`swa_reduce_kernel`，本 patch 已覆盖，Δ256 KiB/实例）。若 spec 流偏移仍在，按同法分别看 draft 与 target 的列 0。

---

## 附：本轮读过的关键位置（便于复核）

- `targets/qwen3_6/impl/runtime/text_context_impl.h:1047-1070`（target 注意力：plain 与 verify **同一个** `ops::gqa_attention` 调用点）
- `wrapper/gqa_attention.cpp:404-412`（路由：`width∈[1,6]→SmallT`，`24Q/width=8→Prompt`）、`:336-346`（partial 分配）、`:473-477`（Prompt 容量=0）、`:567-589`（`gqa_attention_cached` → SmallT 分裂）
- `launcher/gqa_attention_decode.cu:483-503`（split 容量）、`:624-631`（共享 reduce 发射）
- `kernel/gqa_attention_geometry.cuh`（`Gqa27Geometry = <24,4,1>`、`GqaMuseGeometry = <32,2,1,128>`）
- `kernel/gqa_attention_decode.cuh:147-261`（**6 档共用** reduce）、`:98-121`（split 策略）
- `targets/qwen3_6_27b/impl/variant.cpp:23-39`（**默认 KV = NVFP4 + 10 层 E8Kv**）
- `ops/softmax_attention/dense/causal_cache/small_t_fp8.cuh:52,580,595`（**既有的 FP32 partial 参考实现**）
- `src/CMakeLists.txt:67-102`（被编译的 TU：`gqa_attention_decode.cu` / `_e8.cu` / `bidirectional_gqa_attention.cu` / `swa.cu`；**无** `_smallt.cu`/`_g35.cu`/`_muse.cu`）
