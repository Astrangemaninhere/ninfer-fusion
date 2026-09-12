# PART-B：split-K 注意力 `partial_acc` 由 **BF16 → FP32**（统一归约精度档）

树：`/home/user/ninfer-fusion`（**只读**：未编译、未跑引擎、未改任何源文件；见 §8 的 mtime 证据）。
产物：`_collab/PART_B_patch.diff`（唯一 diff）、本报告。
**未改动**路由/分派、`partial_m/l`（本就 FP32）、KV 量化、`partial_acc` 以外的任何缓冲。

---

## 0. 结论先行

| 项 | 结果 |
|---|---|
| 统一 diff | `PART_B_patch.diff`：**29 个文件**，`+215 / −227` 行，63,050 B，md5 `39ca95d1ac379d1dfd8f53be237cb4f5` |
| `patch -p1 --dry-run` | 在 `/home/user/ninfer-fusion` **exit=0**，`checking file` ×29，**0 FAILED hunk**（仅 dry-run，树未变） |
| 残扫（改后） | `__nv_bfloat16* partial_acc` 声明 **0**；`static_cast<…bfloat16*>(partial_acc.data)` **0**；`load_vec<int4>(&qkv_s…)` 写 partial **0**；`__bfloat162float(partial_acc…)` **0**；写点只剩 `= 0.0f`（neutral）与 `make_float2(...)` |
| 寄存器/共享 | **均不增加**。`shared` 对 4 个 staging 型内核**完全不变**（被删的 staging 用的是仍然要留给 Q 暂存的 `qkv_s`），并**少一次 `__syncthreads()`**；`__launch_bounds__` 未动 ⇒ 占用率不变 |
| 最大单点代价 | 每层 partial 缓冲 +Δ（见 §4）；T=1/window=1200 时 Δ=+228 KiB，**相对该层注意力自身 K/V 流量 ≈ +19%**，但相对该 step 的权重流量 ≈ **+0.2%** |
| 最大相对回归风险 | **E 组（swa / bidirectional，T=16/W=4096）**：Δ 往返 +8 MiB vs 该层 K/V 16 MiB ⇒ **+50%**。若 tok/s 退化，**先回退 E 组**（§4.3 给了 5 个文件） |

### 0.1 必须订正的一处（重要，影响"改哪里才有用"）
任务书/R31 把"plain(T=1) 的 split-K"指向 `softmax_attention/dense/context/kernel.cuh`。**代码事实是两套并存的实现，而 live 的那一套不在 dense/context**：

| 实现 | 入口 | 头几何（窄校验） | targets 调用者 |
|---|---|---|---|
| **live：`ops::gqa_attention`** | `src/ops/wrapper/gqa_attention.cpp:342` → `launcher/gqa_attention_decode_smallt.cu` → **`ops/kernel/gqa_attention_decode_bf16.cuh`** + reducer **`ops/kernel/gqa_attention_decode.cuh:247`** | `(24,256)` / `(16,256)` / `(32,128)`（`gqa_attention.cpp:444-448`） | **`src/targets/qwen3_6/impl/runtime/text_context_impl.h:522,527,1061,1066`** |
| 旁路：`ops::context_softmax_attention` | `dense/context/context_softmax_attention.cpp:172` | **只接受 32Q/8KV/D128**（`:22-25`） | 全树 grep **无 targets 调用者**（include/public 声明在 `include/ninfer/ops/softmax_attention.h:182`） |
| 旁路：`ops::causal_softmax_attention` | `dense/causal_cache/causal_softmax_attention.cpp` | `(24,4)` / `(16,2)`（`small_t.cu:247,250`） | 无 targets 调用者 |

27B 的事实（`src/targets/qwen3_6_27b/impl/config.h:37-39`）：`query_heads=24, kv_heads=4, head_dim=256` ⇒ `Gqa27Geometry = GqaGeometry<24,4,1>`（`ops/kernel/gqa_attention_geometry.cuh:25`，`DecodeSplits = 85`）。
`ops::gqa_attention` 的 SmallT 判据：`width∈[1,6] → SmallT`（`gqa_attention.cpp:404`）+ `gqa_attention_uses_small_t(T≤6)`（`launcher/gqa_attention_decode.cu:481` ⇒ **plain T=1 命中**），而 **verify 的 T=8 走 `Prompt`**（无 split；`gqa_attention.cpp:408-412` 的 `q_heads==16` 分支在 24Q 上不成立）⇒ 与 R30 §三的分叉机理一致。
⇒ **本次"精度档对齐"真正起作用的是 A+B+C 组（12 文件）**；D/E/F 组是"统一算术"的一致性 + 用户点名 site，附代价与回退组。

---

## 1. 全量改动清单（行号为**改前**行号）

### 1.1 类型：`partial_acc` 指针声明（`__nv_bfloat16*` → `float*`）
| file:line | 改前 → 改后 |
|---|---|
| `ops/kernel/gqa_attention_decode_bf16.cuh:25` | `__nv_bfloat16* partial_acc, float* partial_m,…` → `float* partial_acc, …` |
| `ops/kernel/gqa_attention_decode_i8.cuh:81` | 同上 |
| `ops/kernel/gqa_attention_decode_nvfp4.cuh:138` | 同上 |
| `ops/kernel/gqa_attention_decode_iso3.cuh:32` | 同上 |
| `ops/kernel/gqa_attention_decode_fp8.cuh:31` | 同上 |
| `ops/kernel/gqa_attention_decode.cuh:149` | `const __nv_bfloat16* partial_acc, …` → `const float* partial_acc, …`（reducer 入参） |
| `ops/kernel/bidirectional_gqa_attention.cuh:116,528,543,553,632,646` | ×6：两个 partial 内核 + `swa_split_partial_kernel` + `noncausal_gqa_reduce_body` + `bidirectional_gqa_reduce_kernel` + `swa_reduce_kernel` |
| `softmax_attention/common/context_query.cuh:95,469` | ×2：`context_query_split_partial_body` / `context_query_reduce_body` |
| `softmax_attention/dense/context/kernel.cuh:77,93` | ×2：两个 kernel 的形参 |
| `softmax_attention/sliding_window/kernel.cuh:47,67` | ×2（复用同一 body / 自带 reduce） |
| `softmax_attention/dense/causal_cache/small_t_bf16.cuh:27` / `small_t_i8.cuh:67` / `small_t.cuh:158` | ×3 |

### 1.2 分配点（`workspace.alloc` dtype）
| file:line | 改前 → 改后 |
|---|---|
| `ops/wrapper/gqa_attention.cpp:342`（**live**） | `DType::BF16, {head_dim, q_heads, tokens, splits * batch_size}` → `DType::FP32, {…}` |
| `ops/wrapper/swa.cpp:68` | `DType::BF16, {kHeadDim, kQHeads, tokens, splits * batch_size}` → `DType::FP32` |
| `ops/wrapper/bidirectional_gqa_attention.cpp:83` | 同上 |
| `softmax_attention/dense/context/context_softmax_attention.cpp:90` | 同上 |
| `softmax_attention/sliding_window/sliding_window_attention.cpp:78` | 同上 |
| `softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:269-270` | `cache_dtype == DType::FP8_E4M3FN ? DType::FP32 : DType::BF16,` + 换行 `{kHeadDim, …}` → `DType::FP32, {kHeadDim, …}`，并在函数首行加 `(void)cache_dtype; // split-K partials are FP32 for every KV tier`（防 unused-parameter 告警） |

> 注：`small_t_fp8.cuh:52,595` / `small_t_fp8.cu:62,87` **本来就是 `float*`**（`causal_softmax_attention.cpp:269` 的旧三元式就是为它准备的）⇒ 本次改动把它从"仅 fp8 档"推广到"全档"，是既有先例的推广而非新发明。这 2 个文件**未被 diff**。

### 1.3 打包存储点（partial 写）
| file:line | 改前 → 改后 | 宽度 |
|---|---|---|
| `common/context_query.cuh:449,461` | `store_vec(&partial_acc[dst], pack_bf16x2(acc[n][0], acc[n][1]))` → `… make_float2(…)` | 4 B → **8 B** |
| `kernel/bidirectional_gqa_attention.cuh:503,515` | 同上（DirectOutput=false 分支） | 4 B → 8 B |
| `kernel/gqa_attention_decode_i8.cuh:802,810` | `*reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(…)` → `*reinterpret_cast<float2*>(…) = make_float2(…)` | 4 B → 8 B |
| `kernel/gqa_attention_decode_nvfp4.cuh:1010,1018` | 同上 | 4 B → 8 B |
| `dense/causal_cache/small_t_i8.cuh:736,744` | 同上 | 4 B → 8 B |
| **128-bit 暂存型 4 处**：`kernel/gqa_attention_decode_bf16.cuh:396-426`、`kernel/gqa_attention_decode_iso3.cuh:536-566`、`kernel/gqa_attention_decode_fp8.cuh:479-509`、`dense/causal_cache/small_t_bf16.cuh:479-509` | `qkv_s[…] = __float2bfloat16(acc[n][j]);`（4 次 smem 存）+ `__syncthreads()` + `store_vec(&partial_acc[dst], load_vec<int4>(&qkv_s[row * D + d]))` → **直接** `store_vec(&partial_acc[dst], make_float2(acc[n][0], acc[n][1]))`（来自寄存器 `acc`） | 16 B → 8 B（每 8 元素：1 → 4 条；见 §3） |

### 1.4 neutral（零填充）写点
| file:line | 改前 → 改后 |
|---|---|
| `gqa_attention_decode_bf16.cuh:102-103`、`_i8.cuh:183-184`、`_iso3.cuh:109-110`、`_fp8.cuh:108-109`、`small_t_bf16.cuh:104-105`、`small_t_i8.cuh:168-169` | `partial_acc[idx] = __float2bfloat16(0.0f);` → `= 0.0f;` |
| `gqa_attention_decode_nvfp4.cuh:250-252` | 宏内续行形式 `…TokenTile)] = \` / `__float2bfloat16(0.0f);  \` → `0.0f;`（保留续行反斜杠） |

### 1.5 reduce 读取点（去 bf16 解包）
`__bfloat162float(\n partial_acc[…] )` → `partial_acc[…]`，共 **6** 处：
`gqa_attention_decode.cuh:247-250`、`bidirectional_gqa_attention.cuh:621-623`（noncausal reduce）、`bidirectional_gqa_attention.cuh:709-712`（swa reduce）、`context_query.cuh:528-531`、`sliding_window/kernel.cuh:130-133`、`dense/causal_cache/small_t.cuh:257-258`。

### 1.6 调用/类型转换点（`Tensor& partial_acc` → kernel 指针）
`static_cast<__nv_bfloat16*>(partial_acc.data)` → `static_cast<float*>(…)`；const 版同理。共 **31** 处：
`launcher/gqa_attention_decode.cu` ×6（`:157,190,225,280,414` 内核 + `:627` reducer）、
`launcher/gqa_attention_decode_smallt.cu` ×1（`:159`）、
`launcher/gqa_attention_decode_e8.cu` ×1（`:60`）、
`launcher/gqa_attention_decode_partial.cuh` ×5（`:166,199,234,289,421`）、
`launcher/gqa_attention_decode_impl.cuh` ×6（`:147,180,215,270,402,566`）、
`launcher/bidirectional_gqa_attention.cu` ×3（`:128,156,162`）、
`launcher/swa.cu` ×3（`:116,135,145`）、
`softmax_attention/dense/context/launch.cu` ×3（`:128,156,162`）、
`softmax_attention/sliding_window/launch.cu` ×3（`:118,137,147`）、
`dense/causal_cache/small_t.cu` ×3（`:120,166,330`）。

> `partial_m/l` 的三处 `static_cast<float*>(partial_m.data)` 全部**未动**（本就 FP32）。
> `out` 的 `static_cast<__nv_bfloat16*>(out.data)` 与 `store_vec(&out[…], pack_bf16x2(…))` 全部**未动**（输出仍是 BF16）。

---

## 2. 分组（便于回退 / 分步验收）

| 组 | 文件数 | 内容 | 是否 live | 单独回退的合法性 |
|---|---|---|---|---|
| **A** | 6 | `ops/kernel/gqa_attention_decode_{bf16,i8,nvfp4,iso3,fp8}.cuh` + `…_decode.cuh` | **是**（plain T=1 SmallT） | 必须与 B、C 同进同退 |
| **B** | 5 | `ops/launcher/gqa_attention_decode{.cu,_smallt.cu,_e8.cu,_partial.cuh,_impl.cuh}` | **是** | 同上 |
| **C** | 1 | `ops/wrapper/gqa_attention.cpp`（dtype 分配） | **是** | 同上 |
| **D** | 5 | `dense/causal_cache/small_t_{bf16,i8}.cuh`、`small_t.{cuh,cu}`、`causal_softmax_attention.cpp` | 否（无调用者） | 自成一体，可整组回退 |
| **E** | 5 | `ops/kernel/bidirectional_gqa_attention.cuh`、`launcher/bidirectional_gqa_attention.cu`、`launcher/swa.cu`、`wrapper/bidirectional_gqa_attention.cpp`、`wrapper/swa.cpp` | **是**（DFlash/DFlash2 草稿路径 `dflash_impl.h:315,322,366`、`dflash2_impl.h:256`） | 自成一体，可整组回退（**性能风险最高的一组**） |
| **F** | 7 | `common/context_query.cuh`、`dense/context/{kernel.cuh,launch.cu,context_softmax_attention.cpp}`、`sliding_window/{kernel.cuh,launch.cu,sliding_window_attention.cpp}` | 否（无 targets 调用者） | 自成一体，可整组回退 |

跨组一致性是**硬约束**：每个家族内同一个 `SmallTWorkspace.acc` / `PartialWorkspace.acc` 被 bf16/i8/nvfp4/iso3/fp8 五种 KV 档共享（`wrapper/gqa_attention.cpp:337-346` 与 `dense/causal_cache/causal_softmax_attention.cpp:264-274` 各自的分配器；`gqa_attention.cu:157,190,225,280,414` 与 `small_t.cu:120,166` + `small_t_fp8.cu:62` 混用同一缓冲）⇒ 只改一半会变成**静默的类型混淆**（C 风格 `static_cast` 于 `void*` 不会报错）。这也是本 patch 必须一次覆盖 5 种 KV 档的原因。

---

## 3. 128-bit 打包存储：为何不能保持，替代方案与判据

**问题**：4 个内核的 partial 写是"先把 `acc[PVNt][4]` 落进 `qkv_s`（bf16）→ `__syncthreads()` → 每线程 `load_vec<int4>` + 16 B 向量存"。BF16 能凑满 128-bit 正是因为 8 个 bf16 = 16 B；FP32 要 8 个 float 才 16 B，而每 lane 只持有 **2 个相邻 d** ⇒ 天然只有 64-bit 粒度。

**为什么不能"把 staging 换成 float"**：`qkv_s` 是 `__shared__ __align__(16) __nv_bfloat16 qkv_s[QkvRows * D]`（`gqa_attention_decode_bf16.cuh:46`，`QkvRows = 2*Bc = 64`，D=256 ⇒ 32 KiB），且内核前半段**还要用它暂存 Q**（`:170-205`）。要暂存 `row_count*D` 个 float 需要 `RowCount*D*4 ≤ Br*D*4 = 32*256*4 = 32 KiB`… 与 bf16 的 32 KiB **恰好等量**，但 Q 暂存与 partial 暂存**同一时刻不复用**（Q 的 ldmatrix 早已消费完），理论上可以把 `qkv_s` 的尾部按 float 重解释——**但容量只有一半（16 KiB）⇒ row 数减半，语义会变**。故放弃。

**采用的替代方案（= 该文件族里既有的量产写法）**：直接由 MMA 片段寄存器写 `float2`（8 B），即 `small_t_fp8.cuh:580,588` / `gqa_attention_decode_i8.cuh:802,810` / `_nvfp4.cuh:1010,1018` **原本就在用的写法**：
```cpp
store_vec(&partial_acc[dst], make_float2(acc[n][0], acc[n][1]));   // row0：d0, d0+1
store_vec(&partial_acc[dst], make_float2(acc[n][2], acc[n][3]));   // row1
```
**判据（为什么这不是"牺牲性能"）**：
1. **指令数净减**：删掉每 n 4 条 bf16 smem 存 + 1 次 `__syncthreads()` + 每 8 元素 1 条 smem load，换成每 n 2 条 8 B 全局存（地址算术与 `row_to_qt` 已在 i8/nvfp4 里证明可接受）。等价写法已在 **i8/nvfp4 两个量产热核** 上跑着。
2. **对齐成立**：`d0 = n*8 + 2*lid` 恒为偶数，且索引步长 `HeadDim*QHeads*…` 为偶数 ⇒ 8 B 对齐满足 `float2` 要求；缓冲是 workspace 首个分配，基址 ≥16 B 对齐。
3. **coalescing**：4-lane 组（`lid=0..3`）覆盖 d0..d0+7 共 **32 B 连续** ⇒ 恰好一个 sector，无浪费；但 warp 内跨 row 不再凑成 128 B。旧 16 B 版同样跨 row 打散，**信噪差异有限**；此项列为必测（§7-2）。
4. **仍然可选（若实测显示 store 吞吐是瓶颈）**：把该 tail 的暂存改为 **动态 smem float 暂存**（`2*KeyBlock*D` 之外再申请 `RowCount*D*4`，bf16 核 D=256 ⇒ +32 KiB，占用率会掉）以换回 16 B 存。预期**不划算**，故不在本 patch。

**结论**：本处属于任务书 §5 的"无法在**不牺牲性能**前提下保持 128-bit"的典型 ⇒ 已给出替代方案（8 B float2，沿用既有量产写法）并给出判据；**不做 128-bit 保持**。

---

## 4. 量化：逐点元素数 / 字节数 / 宽度 / 寄存器与共享

### 4.1 live 路径 A（`Gqa27Geometry`：D=256, QHeads=24, KVHeads=4, GroupSize=6, T=1；`partial_acc` 形状 `{D, QHeads, tokens, splits}`）
`splits`（`gqa_small_t_default_splits` / `gqa_small_t_launch_capacity`，`gqa_attention_decode.cuh:83-96`；`DecodeSplitScale=1`，`DecodeSplits=85`）：

| window | splits | 元素数 | BF16（改前） | FP32（改后） | Δ/layer | 该层 K+V 读 | Δ 往返 / K+V |
|---|---|---|---|---|---|---|---|
| 1 200 | 19 | 116 736 | 228.0 KiB | 456.0 KiB | **+228 KiB** | 4 800 KiB | **+19.0 %** |
| 1 800 | 29 | 178 176 | 348.0 KiB | 696.0 KiB | +348 KiB | 7 200 KiB | +19.3 % |
| 2 048 | 32 | 196 608 | 384.0 KiB | 768.0 KiB | +384 KiB | 8 192 KiB | +18.8 % |
| 4 096 | 64 | 393 216 | 768.0 KiB | 1 536 KiB | +768 KiB | 16 384 KiB | +18.8 % |
| 8 192 | 64 | 393 216 | 768.0 KiB | 1 536 KiB | +768 KiB | 32 768 KiB | +9.4 % |
| 16 390 | 65 | 399 360 | 780 KiB | 1 560 KiB | +780 KiB | 65 560 KiB | +4.8 % |
| 32 768 | 69 | 423 936 | 828 KiB | 1 656 KiB | +828 KiB | 131 072 KiB | +2.5 % |
| ≥65 536 | 85（cap） | 522 240 | 1 020 KiB | 2 040 KiB | +1 020 KiB | 262 144 KiB | +1.6 % |

"Δ 往返"= partial 写 + reduce 读 = 2×Δ。**相对该层注意力自身流量**在短上下文最贵（+19%），长上下文趋近 0；**相对整个 decode step**：权重每 step ≈ 13.5 GB（nvfp4 27B）⇒ Δ 单层 228 KiB ⇒ 全栈（约 60 层）≈ +28 MB/step ≈ **+0.2 %**。⇒ 与 R31 §二"可忽略"的判断一致（量级 2×10⁻³）。

### 4.2 其余组（同法，`workspace.alloc` 形状见 §1.2；分片数取各 plan 的上限）
| 组 | 几何 | T | splits | 元素数 | BF16 | FP32 | Δ |
|---|---|---|---|---|---|---|---|
| D（causal small-T） | D256/24Q/4KV | 1 / 6 | 19（w=1200） | 116 736 / 700 416 | 228 KiB / 1 368 KiB | 456 KiB / 2 736 KiB | +228 / +1 368 KiB |
| D | 同上 | 6 | 64（w=4096） | 2 359 296 | 4 608 KiB | 9 216 KiB | +4 608 KiB |
| F（dense/context、sliding_window） | D128/32Q/8KV，`split_limit ≤32`（T≤8）或 ≤40（T=16, key_block 64） | 8 | 32 | 1 048 576 | 2 048 KiB | 4 096 KiB | +2 048 KiB |
| F | 同上 | 16 | 32 / 40 | 2 097 152 / 2 621 440 | 4 096 / 5 120 KiB | 8 192 / 10 240 KiB | +4 096 / +5 120 KiB |
| E（bidirectional / swa） | D128/32Q/8KV，`split_limit=32`（`launcher/bidirectional_gqa_attention.cu:62`） | 1 / 8 / 16 | 32 | 131 072 / 1 048 576 / 2 097 152 | 256 KiB / 2 048 KiB / 4 096 KiB | 512 KiB / 4 096 KiB / 8 192 KiB | +256 KiB / **+2 MiB** / **+4 MiB** |
| E（相对该层 K/V = `swa` window 4096 ⇒ 16 MiB/call） | — | 16 | 32 | — | — | — | **+8 MiB 往返 = +50 %** ⚠ |

### 4.3 向量化 store 宽度变化一览
| 形态 | 改前 | 改后 | 说明 |
|---|---|---|---|
| `store_vec(…, pack_bf16x2)`（context_query / bidirectional，4 站点） | 4 B（2×bf16） | **8 B（2×f32）** | 指令数不变；字节 ×2 |
| `*reinterpret_cast<unsigned*>`（i8 / nvfp4 / small_t_i8，6 站点） | 4 B | **8 B（float2）** | 指令数不变；字节 ×2 |
| `store_vec(…, load_vec<int4>)`（4 个 128-bit 暂存核） | **16 B（8×bf16）** | **8 B（2×f32）** | 每 8 元素 1→4 条；同时删 4 条 smem 存/元素组 + 1 barrier + 每 8 元素 1 条 smem load ⇒ 总指令数净减 |
| reduce 读 | `__bfloat162float(ld.f32)` = F2F + 1 条 | 直接 `ld.f32` | 每元素省 1 条 CVT；`active_splits×QHeads×T×D` 次 |
| `partial_m/l` | FP32（未动） | FP32 | — |
| `out` 写 | BF16（未动） | BF16 | — |

### 4.4 寄存器 / 共享 / 占用率判断
| 内核 | 寄存器 | 共享 | 占用率 |
|---|---|---|---|
| 4 个 128-bit 暂存核（A-bf16 / A-iso3 / A-fp8 / D-bf16） | **不增**（`make_float2` 直接吃既有 `acc[n][0..1]`；新局部量仅 `prow0/prow1/d0/aq_head/atoken`，`#pragma unroll` 下由 `d0` 线性表达式复算，`n` 展开 32 次但 `d0` 只有一个活值）；**净减**（不再需要 staging 的地址寄存器） | **不变**（被删的 staging 用的是仍留给 Q 暂存的 `qkv_s`；未新增缓冲）；**少 1 次 `__syncthreads()`** | 不变（`__launch_bounds__(128,2)` / `(WarpsPerCta*32,2)` 未动，smem 未增） |
| 3 个 direct-store 核（A-i8 / A-nvfp4 / D-i8） | 不增 | 不变 | 不变 |
| reducer / reduce body（全部） | 不增（删掉 `__bfloat162float`） | 不变（`__shared__ float reduce[128/256]`、`weights[W][128]` 未动） | 不变 |
| `context_query_split_partial_body`（D128 族） | 不增 | 不变 | 不变 |

⇒ **无一处增加寄存器或共享用量，也没有任何缓冲被"升级为大缓冲"**（`partial_m/l` 已 FP32；`out`/KV/Q 全程未动）。

---

## 5. 性能代价汇总（正面与负面都给）

| # | 项 | 方向 | 量级 |
|---|---|---|---|
| 1 | partial 缓冲足迹 | **负** | ×2（§4.1/4.2；live 路径 228 KiB→456 KiB @w=1200） |
| 2 | partial 写 + reduce 读流量 | **负** | 2×Δ（§4.1：+19% 注意力自身流量 @短上下文；+0.2% step 流量） |
| 3 | partial 写指令数（4 个暂存核） | **正** | 减少（删 smem 暂存 + barrier + smem load） |
| 4 | reduce 读指令（去 CVT） | **正** | 每元素 −1 条 |
| 5 | `partial_acc` 写合并度（暂存核） | **负（小）** | 128 B → 32 B/4-lane 组；旧写法同样跨 row 打散，且 i8/nvfp4 量产同形 |
| 6 | 寄存器/共享/占用率 | **中性** | §4.4 |
| 7 | verify（T=8，走 Prompt，**无 split**） | **中性** | 不经过任何被改内核；`Prompt` 路径零改动（`partial_acc` 不参与） |
| 8 | 路由/分派/并行度/网格 | **中性** | 未改任何谓词；grid 形状与 split 数完全不变 |
| 9 | 数值（唯一意图） | **正** | partial 不再经 bf16 舍入；`m/l` 归约顺序不变 |

---

## 6. 残余风险

| 风险 | 评估 | 缓解 |
|---|---|---|
| **未编译**（本轮禁止） | 中：C++ 层面所有改动都是"同形替换 + 一个块级替换"，且 5 种 KV 档同形改完；但模板实例化下的名字遮蔽/`float2` 可见性仍需一次编译确认 | 编译前先看 §7-0 的三条检查；`make_float2`/`float2` 由 `vector_functions.h` 提供，同文件已有 `make_int4` ⇒ 传递包含成立 |
| **E 组回归**（swa/bidirectional，T=16/W=4096 +50%） | **中高**（唯一实质负项） | tok/s 退化时**先整组回退 E 组 5 文件**（§2） |
| 4 个暂存核删 barrier 的语义 | 低 | 已逐个核对：被删块是内核尾部、其后无任何线程再读 `qkv_s`（bf16:427、iso3:567、fp8:510、small_t_bf16:510 均为函数尾）⇒ 无跨线程依赖残留 |
| `d1` 变量删除后的 `-Wunused` | 低 | `d1` 声明已随块删除；`pack_bf16x2`/`load_vec` 若在本 TU 变未使用，它们是 header `__forceinline__`，不触发 unused 告警 |
| `-Wunused-parameter`（`cache_dtype`） | 低 | 已在 `causal_softmax_attention.cpp` 加 `(void)cache_dtype;` |
| tests/bench 的黄金值 | 低 | `tests/ops/softmax_attention/{context,causal_cache}.cpp`、`tests/ops/test_sliding_window_attention.cpp`、`bench/ops/{context_softmax_attention,causal_softmax_attention,sliding_window_attention}_bench.cu` **不直接持有 partial 缓冲**（grep `partial_acc` 于 `tests/ bench/ tools/ examples/ apps/` 无命中），它们经 `*_workspace_capacity_bytes()` 取容量 ⇒ 容量已随之 ×2，不会欠分配；只是若某断言的容差是"对 bf16 版 golden"的，数值会变（应该是**更接近**参考实现） |
| 与 R31 判据的关系 | — | 若 `verify 列0 一致率` 仍停在 69%，则本项"必要但不充分"：核心嫌疑转向 `Prompt` 侧/post-attention（R31 §四已预置该分支） |
| 精度提升的**上限** | 需说明 | 本 patch 消除的是 **split 内 partial 的 bf16 舍入**；plain 与 verify 之间**仍会剩**：split 数随 window 变化的 FP32 结合序差（A 组内 T=1 vs T=8 分别走 SmallT/Prompt 的天然差异）⇒ 判据只能要求"收敛"，不能要求逐位相等 |

---

## 7. 需主代理实测清单（按顺序）

0. **编译前 3 条静态检查**（30 秒）：`grep -rn "__nv_bfloat16\* *\(__restrict__ \)\?partial_acc" src` 应为空；`grep -rn "bfloat16\*>(partial_acc.data)" src` 应为空；`grep -rn "pack_bf16x2(acc\[n\]" src` 应只剩写 `out` 的 DirectOutput 分支。
1. **编译**（唯一编译窗口内的第一件事）：`fused`/核心 TU 若报 `float2`/`make_float2` 未定义，只需在报错文件补 `#include <vector_functions.h>`（预期不需要）。
2. **逐列一致率（R31 §三判据）**：`verify 列 0 与 plain 一致率` **69% → 期望 ≥90%**（样本 ≥29，同 R29 的 token 锚定探针）。同测列 1（75% → 期望 ≥90%）。
3. **tok/s 硬底线**：plain decode tok/s 与 verify/spec tok/s **均不得退化**。若退化，按 §2 分组二分：先回退 **E 组**（最贵），再回退 **D/F 组**（非 live），保留 **A+B+C**（live 且指标受益方）。
4. **DFlash/DFlash2 路径**（E 组的真实使用者 `dflash_impl.h:315,322,366`、`dflash2_impl.h:256`）：单测其接受率与耗时——E 组是唯一把 partial 往返推到 +50% 相对流量且**位于投机草稿关键路径**的地方。
5. **首次偏离点**：`zh plain vs dflash2` 首次偏离位置（R31 §三-2），期望后移/消失。
6. **长上下文回归**：window ∈ {4096, 32768} 各跑一次（partial 足迹在 4096 处 1.5 MiB/layer，是"分配器压力"的峰值点）。
7. **验收后**：再落 R31 §四-2 的"暂存 TU 拆分"，此后这类"改精度档 + 复测"的迭代成本才能从半小时级降到分钟级。

---

## 8. 纪律与证据

- **只读**：本轮未编译、未跑引擎/GPU、未 `find` 全盘。
- **不改构建树**：全程只做 `patch -p1 --dry-run`（exit=0）。旁证：`find /home/user/ninfer-fusion/src -newermt '-40 minutes' -printf '%TT %p\n' | sort` 只列出 **18 个与本 patch 无关的文件**（`attn_input_proj/*`、`gdn_*`、`linear*` 的 plan/config、`targets/.../text_context_impl.h`），时间戳为 `16:38:26` / `16:42:23`——**均早于本会话**（会话 17:07 起），属并行 writer 的改动；**我改动的 29 个文件一个都没出现在该列表里**（`-25 minutes` 窗口为空）。
- **diff 形式**：`diff --git a/<rel> b/<rel>` + `--- a/…` / `+++ b/…` + 标准 hunk ⇒ `patch -p1` 可直接应用（`gqa_attention_decode_impl.cuh` 是 **CRLF** 文件，其 hunk 内保留了 `\r`，这是它能干净 dry-run 的原因；**不要**对该文件做换行归一化）。
- 因 `/tmp` 会被 WSL 重启清空，本轮所有中间产物（scratch 副本、转写脚本、核验脚本）均**未**留在 `_collab`；本报告与 `PART_B_patch.diff` 自包含。
