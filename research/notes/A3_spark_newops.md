# A3 · Spark-X2.5 三项 new_op 补丁草案（只生成，不落地）

- 日期：2026-09-10
- 范围：**未改 `src/**`、未改 `tools/**`、未编译、未开 nvcc/ptxas、未用 GPU**。只做文本读 + 生成 diff + 在**影子树**上 `patch -p1 --dry-run`。
- 基线：**活体构建树 `/home/user/ninfer-fusion`**（不是 S54 用的镜像 `ninfer-fusion-repo`）。原因见 §1 —— 两棵树在本次要动的文件上有 6 处漂移，镜像行号不可直接引用。
- 三个补丁产物：
  - `_collab/A3_spark_head_geometry.diff`（new_op #1，8 文件，+140/-21，335 diff 行）
  - `_collab/A3_spark_headwise_gate.diff`（new_op #2 路线 b，6 文件，+153/-5，219 diff 行）
  - `_collab/A3_spark_gelu_mul.diff`（new_op #3，8 文件（6 新增），+384/-0，427 diff 行）
- 生成器：`_collab/A3_mkpatch.py`（每个锚点断言出现次数，漂移即报错而不是产出"看起来对"的补丁）；活体文件字节副本在 `_collab/a3_scratch/live/`，dry-run 影子树在 `_collab/a3_scratch/shadows/`，dry-run 原文在 `_collab/a3_scratch/dryrun_*.txt`。

---

## 0. 结论摘要（先看这 8 条）

1. **16Q/4KV@256 不在 kernel 实例化域内**，而且拦路的不是"缺一个 alias"：`gqa_attention_decode.cu` 的 i8/NVFP4 行 tile 调度表按 group 硬编码，`GroupSize==4` 走"else"臂时**要么静默少算 PV tile，要么直接 static_assert 编不过**（§2.2 表，含逐 TT 数字）。这是 S54 §3.9 R2 的"待证实"项的**证实**。
2. `kv_heads_for_q_heads()`（`src/ops/wrapper/gqa_attention.cpp:25-30`）必须**从"按 q 头数反推"改成"校验 (q_heads, head_dim, kv_heads) 三元组、KV 头数一律取 cache/append 输入"**。16Q/2KV 与 16Q/4KV 在 head_dim 256 上**只能靠 KV 头数区分**，反推法在信息论上就不可能对（§2.3）。
3. 分派点不止 S54 列的 4 处：活体树里凡是 `q_heads==16 && head_dim==256` 的调用都会**落进 35B（16/2）实例**，这样的点有 **4 个**——`gqa_attention_small_t_launch`、`gqa_attention_prompt_attention_launch`、批 `gqa_attention_prompt_launch`，以及 **`gqa_attention_cached_small_t_launch` 的尾部（那里连检查都没有，是纯粹的 fall-through）**；另有 **2 个 E8 TU**（`decode_e8.cu` / `prefill_e8.cu`，S54 未提）用 `q.ne[1] != 16 || q.ne[0] != 256` 做检查、同样把 16/4 放行到 35B。补丁 A 把这 6 个点一起收紧（不收紧就是"16/4 静默当 16/2 跑"）。
4. 补丁 A 的**策略**：BF16 / FP8 / ISO3 档**接受** group 4（这三个档的小 T 内核按 `(TOKENS, WARPS)` 取 `WarpsPerCta`，与 group 无关，静态断言只要求 `Wc*16 ≥ tokens*GroupSize`，group 4 全部满足）；**i8 / E8 / NVFP4 档显式拒绝**（`require_group4_schedule`，`[[noreturn]]`，附原因与替代做法），等它们各自的 group-4 调度表被实测后再开。v1 = BF16 权重 + BF16 KV + T≤512（S54 §7）不受影响。
5. **逐头门控（new_op #2）**：路线 (a)（转换期把 `g_proj` 展成 `[4096,2560]`，复用现有 `sigmoid_mul`）**引擎侧零改动**——`text_attention_projection` 本来就按 `query_size` 行分配 gate（`workspace_recipe.h:62-63`），`attn_mix` 本来就 `view({head_dim,n_q,T})` + `sigmoid_mul(gate,a)`（`text_context_impl.h:969,1071`），且 `(4096,2560)` 这个 linear 几何 Spark 的 q 投影**本来就要注册**；代价是 **+755 MB BF16 权重（+9.2% 权重流量/显存）**，落在 `tools/convert` 侧（本任务不写 tools）。
6. 路线 (b) 已落成 `A3_spark_headwise_gate.diff`：**扩展现有 `sigmoid_mul` 家族**（同一符号、同一签名，加一条"逐头标量门"形状分支 + 1 个 kernel + 1 个 launcher），**不新增 op 文件、不动 CMake**；额外权重 ≈ 0（16×2560×2B×36 = **2.95 MB**），代价是 6 文件 +153 行。已验证**与既有调用不重叠**：活体树只有 3 个调用点（`text_context_impl.h:531,672,1071`），全部传 `[head_dim, n_q, T]` 的 gate，而新分支要求 `gate.ne[0] == x.ne[1]`（即 `n_q == head_dim`），三种几何都不满足（16≠256, 24≠256, 32≠128）⇒ 对 27b/35b/muse **零行为变化**。
7. **gated GELU（new_op #3）**：`ops::gelu` 是 in-place 单元激活、全 op 清单里没有任何"两输入逐元素乘"（只有 `silu_mul`/`sigmoid_mul`/`residual_add`/`linear_add`），所以必须新增一个 op。补丁 C 照 `silu_mul` 五件套加 `ops::gelu_mul(gate, up, out)`（精确 erf，复用 `ops/kernel/gelu.cuh` 的 `gelu_one<false>`），**对现有模型零影响**：改动只有 2 行 CMake + 6 个新文件，既有叶子/既有 op 一行未动（§4.4 给了三条可复跑的证明路径）。
8. 三个补丁各自 `patch -p1 --dry-run` **rc=0**（原文见 §5），且落地动作全部发生在 `_collab/a3_scratch/shadows/`；活体树 36 个相关文件会话前后 md5 **逐字节相同**（`TREE_UNCHANGED`，§6.2）。

---

## 1. 口径与基线：活体树 vs 镜像（S54 的行号在这一批文件上不可用）

S54 的行号口径是镜像 `ninfer-fusion-repo`。本次先做了一次 md5 对照（36 个相关文件，`_collab/a3_scratch/cmp_md5.py` + `live_md5.txt`）：

| 文件 | 镜像 md5 | 活体 md5 | 差异内容 / 对本任务的影响 |
|---|---|---|---|
| `src/ops/launcher/gqa_attention_decode.cu` | `e96962e7…` | `b0a130e2…` | 活体=**单 TU**（S23 的 TU 拆分只进了镜像）+ **S45d 的 nvfp4 `if constexpr` 几何守卫**（活体 `:532,548,557`）⇒ 分派点数量/行号全不同 |
| `src/ops/launcher/gqa_attention_decode_impl.cuh` | `a9e0b273…` | `8e55b778…` | 活体仍是 CRLF 旧版；镜像那份是 S23 重生成的 |
| `gqa_attention_decode_{bf16,i8,fp8,iso3,nvfp4}.cu` | 存在 | **不存在** | 活体**未应用** C_s23 的 TU 拆分 ⇒ S54 §3.9 列的"每个档 TU 各 2-3 处分派"在活体上其实是**同一个 TU 内的 6 个分支** |
| `src/ops/kernel/gqa_attention_decode.cuh` | `21f3edad…` | `d7ed2778…` | 活体**已含 S36 的 `Geometry::HeadDim` 泛化**（`:21-24` 的 S36 注释 + `:37`），镜像没有 ⇒ 头维已不是缺口 |
| `src/ops/launcher/gqa_attention_prefill.cu` | `b2da4c43…` | `50e265e6…` | 活体多 31 行（S45d 的 7 处 `if constexpr` 守卫） |
| `src/CMakeLists.txt` | `41869c87…` | `19c515e9…` | 源文件清单不同（TU 拆分） |
| 其余 26 个（含 `gqa_attention.cpp:25-30`、`sigmoid_gate_mul.*`、`test_sigmoid_mul.cpp`、`bf16_dispatch.cpp`、`tests/CMakeLists.txt`…） | — | — | **SAME**（S54 对这些文件的引用仍然有效） |

⇒ **本批 diff 全部按活体树生成**；§2/§3 的引用行号也全部是活体树行号（镜像行号我会在括号里标 `[mirror:…]`，只在两者都有该文件且都 SAME 时省略）。

---

## 2. new_op #1：主注意力头几何 16Q/4KV@256

### 2.1 三个子问题（分开回答，别混）

**(a) 注册**：`src/ops/kernel/gqa_attention_geometry.cuh:25-27` 只有
`Gqa27Geometry=<24,4,1>` / `Gqa35Geometry=<16,2,2>` / `GqaMuseGeometry=<32,2,1,128>`。
Spark 需要 `GqaGeometry<16,4,S,256>`。**`S`（DecodeSplitScale）取 1**，判据不是口味：
decode 的 grid 是 `dim3(Geometry::KVHeads, splits, batch)`（活体 `gqa_attention_decode.cu:140/169/202/252/366`），
`DecodeSplits = 85*S`（`geometry.cuh:21`），所以一次 launch 的 CTA 数 ≈ `KVHeads * 85 * S`：
27B（4 KV 头, S=1）= 340，35B（2 KV 头, S=2）= 340。Spark 是 4 KV 头 ⇒ `S=1` 与 27B 同量纲（340）。
副产物：`gqa_small_t_split_count/upper_bound` 的所有分段阈值都是 `X/S`，S=1 时**与 27B 完全同值**，所以 `splits` 的选择逻辑不需要新分支。

**(b) 反推 KV 头必须改**：`src/ops/wrapper/gqa_attention.cpp:25-30`
```cpp
std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }   // ← Spark 16Q/4KV 在这里被判成 2KV
    ...
```
`q_heads==16 && head_dim==256` 同时是 35B（16/2, group 8）和 Spark（16/4, group 4），**q 头数不含区分信息**。
所以：KV 头数只能取 `cache.num_kv_heads`（A1/A3 路径）或 `k.ne[1]`（A2 append 路径），函数本身退化为"**这个三元组注册过吗**"的校验。
补丁 A 把它换成 `registered_kv_heads(q_heads, head_dim, kv_heads, op)`，4 个调用点全部改为传 `cache.head_dim, cache.num_kv_heads`（唯一例外是 workspace 容量查询，它拿不到 cache，见 §2.3 的说明）。
调用点：`:254`（A3 单序列校验）、`:288`（A1 批校验）、`:425`（workspace 容量）、`:485`（A1 主体，用于 `require_shape(k/v)`）。

**(c) 实例化域：16Q/4KV@256 落在哪、需要动哪些 TU/宏**

先给结论：**BF16/FP8/ISO3 档落在域内（新实例化即可），i8/E8/NVFP4 档落在域外且不能靠"多一个 alias"解决**。

内核本体是 `template <typename Geometry>`，`GroupSize` 只以**算术**形式出现（`gqa_attention_decode.cuh:76-77,140-142` 的 `q_head = kv_head*GroupSize + local_q`；`decode_bf16.cuh:64` 的 `row_count = tokens*GroupSize`；prefill 三个内核只有 `kv_head = q_head/GroupSize`，活体 `prefill_bf16.cuh:134`、`prefill_i8.cuh:408`、`prefill_nvfp4.cuh:1076`）。head_dim 在活体里已是 `Geometry::HeadDim`（S36）且 `kGqaHeadDim=256`（`decode.cuh:22`）、`kGqaPrefillHeadDim=256`（`prefill_common.cuh:19`）⇒ **256 天然吻合**。

**(c-1) BF16 小 T 档（v1 主路径）= 可用。**
`gqa_attention_decode_bf16.cuh:26-44` 的约束是 `static_assert(QkvRows >= Br)`，`QkvRows = 2*Bc = 64`、`Br = WarpsPerCta*16`，而 `WarpsPerCta` 来自 `(TOKENS, WARPS)` 表（`decode.cu:585-606`：T1→2, T2..T6→4 ⇒ Br = 32/64）。
`row_count = tokens*GroupSize`：group 4 时 T=1..6 → 4/8/12/16/20/24，**全部 ≤ Br**（group 8 时 8..48 也都在域内，所以这个表本来就是 group 无关的）。`gqa_attention_resolve_route`（`:392-401`）里的 `q_heads==16` 分支只决定 SmallT/ChunkedSmallT 路由，不涉及 group。

**(c-2) FP8/ISO3 档 = 可用**（同一 `(TOKENS, WARPS)` 表 + 同两条静态断言；`decode_fp8.cuh:32-33,50`、`decode_iso3.cuh:33-34,51`）。

**(c-3) i8/E8/NVFP4 档 = 域外，且是"编不过或静默错"，不是"慢一点"。**
这两档的 `Wc`（CTA 内 warp 数）按 group 分支硬编码（`decode.cu:285-337` i8、`:417-438` nvfp4、`decode_e8.cu:63-110`），度量是
`consumer_coverage = (Wc / RowTiles) * PVNtPerWarp * 8`，必须 == `head_dim = 256`，
其中 `RowTiles = ceil(TokenTile*GroupSize/16)`、`PVNtPerWarp = 256/((Wc/RowTiles)*8)`，
且有 `static_assert(RowTiles ∈ [1,6])`、`static_assert(Wc % RowTiles == 0)`、`static_assert(PVNtPerWarp ∈ {2,4,8,16})`。

| 档 | TT | RowCount=TT*4 | RowTiles | Wc（group 4 落到的臂） | Consumer | PVNtPerWarp | 覆盖 | 结局 |
|---|---|---|---|---|---|---|---|---|
| i8 | 1 | 4 | 1 | 8（`else` 臂） | 8 | 4 | 256 ✓ | 可编译，正确 |
| i8 | 2 | 8 | 1 | 8 | 8 | 4 | 256 ✓ | 可编译，正确 |
| i8 | 3 | 12 | 1 | 8 | 8 | 4 | 256 ✓ | 可编译，正确 |
| i8 | 4 | 16 | 1 | 16（TT==4 臂） | 16 | 2 | 256 ✓ | 可编译，正确 |
| i8 | 5 | 20 | 2 | **24 / 12 / 12 / 6**（TT==5 的 else 臂，四档 window 全被实例化） | 12 / 6 / 6 / 3 | 2 / **5** / **5** / **10** | 192 ✗（Wc=24 档）/ 其余档 240 ✗ | **Wc=12/6 两档的 PVNt=5/10 违反 static_assert ⇒ 编译失败**（编译期不取决于运行期 window）；若把断言放宽，Wc=24 那档仍**静默少算 8 个 PV n-tile** |
| i8 | 6 | 24 | 2 | 24 / 12 / **6** | 12 / **6** / **3** | 2 / **5** / **10** | 192/240/240 ✗ | **PVNt=5,10 违反 static_assert ⇒ 编译失败** |
| nvfp4 | 1..4 | ≤16 | 1 | 16 | 16 | 2 | 256 ✓ | 可编译，正确 |
| nvfp4 | 5 | 20 | 2 | 12 | **6** | **5** | — | **static_assert 失败 ⇒ 编译失败** |
| nvfp4 | 6 | 24 | 2 | 12 | **6** | **5** | — | **static_assert 失败 ⇒ 编译失败** |

对照（证明我没有读错表）：27B（G6）TT=6 → RowTiles=3, Wc=12 → Consumer=4, PVNt=8 → 256 ✓；35B（G8）TT=5 → RowTiles=3, Wc=24 → Consumer=8, PVNt=4 → 256 ✓。**它们能跑正是因为 RowTiles 恰好把 Wc 分掉且 PVNt 落在 {2,4,8,16} 里；group 4 打破了这个巧合。**

要动的 TU/宏（活体树）：
1. `src/ops/kernel/gqa_attention_geometry.cuh` —— 加 alias。
2. `src/ops/wrapper/gqa_attention.cpp` —— 三元组反推改校验（4 个点）。
3. `src/ops/launcher/gqa_attention.h` —— 加共享拒绝函数 `require_group4_schedule`（+ `<stdexcept>`/`<string>`）。
4. `src/ops/launcher/gqa_attention_decode.cu` —— i8/NVFP4 调度表各加一个 `GroupSize==4` 拒绝臂；两个小 T 分派入口（`:679-703`、`:705-735`）加精确三元组分支；`:479-498` 容量分派加注释。
5. `src/ops/launcher/gqa_attention_decode_e8.cu` —— 调度表拒绝臂 + 2 个分派入口的 16/4 显式拒绝。
6. `src/ops/launcher/gqa_attention_prefill.cu` —— 3 个分派点（`:398-421` 单序列 attention、`:423-442` 单序列 append、`:444-482` 批）。
7. `src/ops/launcher/gqa_attention_prefill_e8.cu` —— 2 个 attention 分派点的 16/4 显式拒绝（append 两点不动，理由见 §2.5）。
8. `include/ninfer/ops/gqa_attention.h:70` —— 注册几何文档行。

> 注意（不改，但要写进评审意见）：`gqa_attention.cpp:263` 与 `:305` 的 `cache.num_kv_heads != kv_heads` 在补丁后**恒为假**（两边同源）。保留它是有意的（防御性；且它是唯一一处"cache 与 k/v 张量一致性"的显式断言）。若评审要求删，删掉即可，无行为差异。
> 也不改：`gqa_kv_append_launch`（prefill.cu:423-442）在活体里对 4 KV 头**本来就落到 `Gqa27Geometry`**，而 fill 内核只读 `Geometry::KVHeads` 与 `Geometry::HeadDim`（`prefill_bf16.cuh:33,44-47,53`）⇒ 对 4/256 平面**数值正确**。补丁只给它补 `&& cache.head_dim == 256`（把一个"碰巧对"变成"证明对"，并顺手堵住"4 KV 头 + 非 256 头维"会误用 256 内核的旧洞）。

### 2.2 补丁 A：`A3_spark_head_geometry.diff`（逐文件）

```
src/ops/kernel/gqa_attention_geometry.cuh                 +5/-0   新 alias + 依据注释
src/ops/wrapper/gqa_attention.cpp                         +33/-9  registered_kv_heads（三元组）+ 4 个调用点
src/ops/launcher/gqa_attention.h                          +15/-0  require_group4_schedule + 头文件
src/ops/launcher/gqa_attention_decode.cu                  +48/-8  i8/nvfp4 拒绝臂 + 2 个精确三元组分派 + 容量注释
src/ops/launcher/gqa_attention_decode_e8.cu               +11/-1  调度表拒绝臂 + 2 个入口拒绝
src/ops/launcher/gqa_attention_prefill.cu                 +20/-2  3 个分派点（16/4 臂 + 35 臂精确化 + append 平面等价注释）
src/ops/launcher/gqa_attention_prefill_e8.cu              +6/-0   2 个入口拒绝
include/ninfer/ops/gqa_attention.h                        +2/-1   注册几何文档
合计 8 文件 +140/-21（335 diff 行）
```

关键 hunk（原文见 diff 文件）：

```diff
-    if (q_heads == 16) { return 2; }
+// A registered GQA geometry is the exact (q_heads, head_dim, kv_heads) triple. ...
+std::int32_t registered_kv_heads(std::int32_t q_heads, std::int32_t head_dim,
+                                 std::int32_t kv_heads, const char* op) {
+    const bool registered = (q_heads == 24 && head_dim == 256 && kv_heads == 4) ||
+                            (q_heads == 16 && head_dim == 256 && kv_heads == 2) ||
+                            (q_heads == 16 && head_dim == 256 && kv_heads == 4) ||
+                            (q_heads == 32 && head_dim == 128 && kv_heads == 2);
```

```diff
-    if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim) {
+    // 16 q-heads at head_dim 256 is ambiguous by q-heads alone: only the KV head count
+    // separates Spark-X2.5 (16/4, group 4) from the 35B geometry (16/2, group 8).
+    if (q.ne[1] == Gqa16x4Geometry::QHeads && q.ne[0] == Gqa16x4Geometry::HeadDim &&
+        cache.num_kv_heads == Gqa16x4Geometry::KVHeads) { ... }
+    if (q.ne[1] == Gqa35Geometry::QHeads && q.ne[0] == Gqa35Geometry::HeadDim &&
+        cache.num_kv_heads == Gqa35Geometry::KVHeads) { ... }
```

```diff
-    if constexpr (Geometry::GroupSize == 16) {
+    if constexpr (Geometry::GroupSize == 4) {
+        require_group4_schedule("int8");          // 或 "nvfp4" / "e8"
+    } else if constexpr (Geometry::GroupSize == 16) {
```
（`if constexpr` 让被丢弃的分支**不被实例化** ⇒ i8/nvfp4 的 static_assert 不会触发；这不能用"运行期 throw"代替，因为编译期实例化照样发生 —— 这是本补丁最容易做错的一点。）

### 2.3 代价

| 项 | 量 | 依据 |
|---|---|---|
| 新增内核实例 | BF16 档 6(TT)×4(batch×masked) + FP8/ISO3 各 6×4 + prefill 5 档 ×2(masked) | `decode.cu:585-606` 的 `switch` 展开 × `launch_profile` 4 组合 |
| 编译时长/内存 | 新增约 **+33% 的 decode TU 内核实例**（原 3 几何 × 24 = 72 → 4 几何 = 96；i8/nvfp4 的 group-4 分支**不实例化**，所以最重的两个档不增长） | S23 的注释实测：i8 档 ≈330 实例是最重的 |
| 运行期 | 0（只是多一个分支判断） | — |
| 拒绝路径 | 用户若对 Spark 用 `--kv-dtype i8/e8/nvfp4`，会在第一次 decode 时抛 `...no measured schedule for q/kv group 4...` | `require_group4_schedule` |

### 2.4 验证方法（按可独立复跑排序）

1. **编译期**（必须）：`make ninfer -j1` 转绿。这一步本身就是 group-4 静态断言的验证 —— 如果 i8/nvfp4 拒绝臂没生效，会在 `static_assert(PVNtPerWarp == 2||4||8||16)` 处失败（§2.1(c-3) 表）。
2. **分派单测**（无需 GPU 数据，只需 CUDA 设备）：对 4 个头数组合各跑一次 `gqa_attention_cached_small_t_launch` 入口：
   `(16,256,2)`→35B 实例（现有行为）、`(16,256,4)`→16x4 实例、`(32,128,2)`→Muse、`(24,256,4)`→27B；未注册的三元组（如 `(16,128,2)`、`(16,256,8)`）必须抛。判据：抛错串里的 `q-heads / kv-heads at head_dim` 三元组与传入一致。
3. **数值对拍**（GPU 窗口）：新几何的 oracle 对拍，口径用公开契约里的 FP64 ideal（`include/ninfer/ops/gqa_attention.h:20-47`）：随机 q/k/v + 4 KV 头缓存，单层 GQA 输出 vs FP64，**误差 ≤1e-2**；同时跑一份 `(16,256,2)` 作为负对照 —— **两者的输出必须显著不同**（若相同，说明 group 被当成了 8，即静默错仍在）。
4. **端到端**：Spark 短 prompt（T≤512）与 HF greedy 逐 token 一致（S54 §4.1 的 logits ≤1e-2 + 逐层 hidden cos ≥0.999）。
5. **接线有效性**：把 `cache.num_kv_heads` 人为改成 4（配 4 头 k/v）后，`q_heads=16` 必须走 16x4；把 k/v 头数改成 2 时必须走 35B —— 这是"不混为一谈"的直接判据。

### 2.5 风险

| # | 风险 | 级别 | 依据/处置 |
|---|---|---|---|
| A1 | i8/NVFP4 的 group-4 调度表**留空**（拒绝而非实现）⇒ 首版 Spark 只能 BF16 KV | 中 | 与 S54 §7 的 v1 范围一致；若日后要 `--kv-dtype nvfp4`（SWA 唯一现成通路），必须先补 group-4 的 `Wc` 表并实测：i8 的可行解是 `Wc=8` 恒定（TT1..6 全满足 `Wc%RowTiles==0` 且 PVNt∈{4,8}），nvfp4 需要重推（RowTiles=2 时 Consumer 必须 ∈ {4,8,16} 且 RowTiles*Consumer ≤ Wc） |
| A2 | 16/4 与 16/2 的**分派靠 cache.num_kv_heads**，若将来出现"q/kv 头数不与 cache 同源"的调用路径（例如草稿注意力自己造 view），收紧后的分支会抛错而不是跑 | 低 | 现状只有 wrapper 会调这些入口（已核对：`grep` 只有 `src/ops/wrapper/gqa_attention.cpp` 的 6 个调用点；`tests/muse128_repro{2,4}.cu` 是未注册的临时复现程序，且它们显式设了 `cache.num_kv_heads`） |
| A3 | 新增约 24 个 BF16 内核实例 ⇒ 单 TU ptxas 峰值上升，与"31.4 GB 内存 + 串行构建"的现状冲突 | 中 | S23 已把 TU 拆过（镜像里），活体没有 ⇒ 建议 A 与 C_s23 的 TU 拆分**同窗落地**，否则 decode TU 的编译内存会回到拆分前水平 |
| A4 | `require_group4_schedule` 放在 `gqa_attention.h`（私有 launcher 头）里是 `inline` + `<stdexcept>`，会进入所有 include 它的 TU | 低 | 只有 3 个 TU include 它；若要更干净可单独开 `gqa_attention_group.h` |
| A5 | 未验证：group 4 下 BF16 小 T 内核的**性能**（Br=32/64、每 warp 16 行，group 4 时 warp 利用率与 27B/35B 不同） | 低-中 | 数值正确性与性能无关；性能按 S54-6 单独测 |
| A6 | `prefill_e8.cu` 的 append 两点**不拒绝** 16/4（与 decode_e8 不一致） | 低 | 理由：append 只写 4/256 平面，与 group 无关，拒绝它反而会让"E8 只做 prefill"这种用法无谓失败；真正的闸门在 E8 decode |

---

## 3. new_op #2：逐头输出门（headwise output gate, sigmoid）

### 3.1 算式与现状（两边都钉死）

HF（`models/Spark-X2.5-4B/modeling_spark.py:154,174,179-180,198-206`）：
```
gate_score = g_proj(h)                      # [b, s, n_q]      n_q = 16
gate_score = gate_score.view(b,s,16,1).transpose(1,2)   # [b,16,s,1]
gate = sigmoid(gate_score.float()).to(attn_output.dtype)
attn_output = attn_output * gate             # 广播到 head_dim
```
引擎（活体 `text_context_impl.h:969,1071` + `workspace_recipe.h:62-63`）：
```
gate = projection.gate.view({head_dim, n_q, T});   // 与 q 同形 = 逐通道门
ops::sigmoid_mul(gate, a, s);                      // 契约：gate 与 x 同形
```
`ops::sigmoid_mul` 的同形要求：`include/ninfer/ops/sigmoid_mul.h:9-20` + `src/ops/wrapper/sigmoid_mul.cpp:36-40`。

### 3.2 路线 (a)：转换期展开 g_proj（零算子）

**引擎侧 diff 为空**。三处现成条件正好拼上：
1. gate 缓冲本来就按 `query_size` 行分配（`workspace_recipe.h:62-63`，与 query 同形）；
2. `attn_mix` 本来就按 `{head_dim, n_q, T}` 视图喂给 `sigmoid_mul`（`text_context_impl.h:969,1071`）；
3. 展开后的 `g_proj` 是 `[n_q*head_dim, hidden] = [4096, 2560]`，**与 Spark 的 q 投影同形** ⇒ 这个 `(4096,2560)` 的 BF16 linear 几何是 S54-2 本来就要注册的（q 需要），不是额外开销。

转换期要做的事（`tools/convert/spark_x2_5_4b/recipe.py`，本任务不写 tools，只描述）：
```
expanded[r, :] = g_proj[r // head_dim, :]     r ∈ [0, 4096),  head_dim = 256
```
即行 `h*256 + c` 复制 g_proj 的第 h 行。**为什么行序是 `h*head_dim + c`**：引擎 `q.view({head_dim,n_q,T})` 的元素 `(d,h,t)` 落在 `d + head_dim*(h + n_q*t)`（活体 `text_context_impl.h:968` 的同一约定），也就是 linear 的第 `r = d + head_dim*h` 行 ⇒ 与 HF `view(...,num_heads,head_dim)` 的 `h*head_dim + d` 逐位一致。**展开公式与视图公式必须一起评审，这是路线 a 唯一的"容易错但不是 diff"的点。**

**代价（算术依据）**
- 常量：`n_q*head_dim*hidden*2B = 16*256*2560*2 = 20.97 MB/层`，36 层 = **754.97 MB**（与 S54 §3.4 的 755 MB 一致）。
- 权重总量（BF16）：每层 `qkv 6144*2560=15.73M` + `g 0.041M` + `o 2560*4096=10.49M` + `mlp 3*10240*2560=78.64M` = 104.90M ⇒ ×36 = 3.776B；加 embedding 131072*2560=335.5M ⇒ **≈4.11B 参数 ≈ 8.23 GB**。
  ⇒ 展开后 **+9.2% 权重体积与解码期权重流量**（decode 是权重带宽瓶颈，等于同比例掉 tok/s），artifact 也 +755 MB。
- 算力：每 token 每层多 `2*4096*2560 = 21.0 MFLOP` ⇒ 0.755 GFLOP/token（36 层）。对 T=512 的 prefill 是多 386 GFLOP，占全模型（≈7.7 TFLOP@T=512）约 5%，可忽略。**瓶颈是显存不是算力。**
- 转换代价：读 36×16×2560×2B = 2.95 MB，写 755 MB（纯复制，无 GEMM）。

**验证**
1. 转换器自检（纯 CPU/盘）：对每层随机抽 256 行做 `expanded[r] == g_proj[r // 256]` 逐字节比较 + 全量 sha256 复核（可离线跑，不需要 GPU）。
2. 引擎 vs HF logits（T≤512）≤1e-2，逐层 hidden cos ≥0.999（S54 §4.1）。
3. **接线有效性负对照**：把某个 head 的 g_proj 行整体置 0（sigmoid→0.5）后 logits 必须显著变化；若不变，说明 gate 没接上（或接错 head）。这条能同时抓"head 映射错"。

**风险**
- 显存/带宽 +9.2%（R8 的核心）；755 MB 的冗余行会随量化档缩小但不会消失（NVFP4 约 210 MB）。
- 转换器一旦把展开做错（行序按 `c*16+h` 而不是 `h*256+c`），引擎会**静默跑出错误的头门**（不再有任何形状检查兜住）⇒ 第 1 条验证是必须的。

### 3.3 路线 (b)：新路由（逐头广播门）—— 已落 `A3_spark_headwise_gate.diff`

设计选择：**扩展现有 `sigmoid_mul` 家族**（同符号、同签名、加一条形状分支），而不是新开 op。
理由：① 变换的数学形式与 `sigmoid_mul` 完全一致，只是 gate 的轴少一维（`ops` 规则 §3"一个头可以放紧密相关的重载/变体"、§8"优先扩展现有家族"）；② 不新增文件、不动 CMake ⇒ 对既有模型的改动面是**同一个文件里的新增分支**；③ 新开 op 需要 5 个新文件 + 2 行 CMake + 新测试注册，收益只是命名更漂亮。

改动清单：

```
include/ninfer/ops/sigmoid_mul.h           +18/-5  契约扩展（两种形态 + 形状判据 + 互斥论证），签名不变
src/ops/wrapper/sigmoid_mul.cpp            +13/-0  headwise_gate_shape() + 早返回分支（既有逐元素代码一行未动）
src/ops/launcher/sigmoid_gate_mul.h        +5/-0   新 launcher 声明
src/ops/launcher/sigmoid_gate_mul.cu       +30/-0  headwise_sigmoid_gate_mul_launch（bf16x8 快路 + 标量回退）
src/ops/kernel/sigmoid_gate_mul.cuh        +32/-0  两个 kernel
tests/ops/test_sigmoid_mul.cpp             +55/-0  5 个 headwise 用例 + 1 个同形方阵回归用例
合计 6 文件 +153/-5（219 diff 行）
```

契约（写入头文件的原话要点）：
```
per-element:     x is [D,H,T], gate is [D,H,T] (same shape)
headwise scalar: x is [D,H,T], gate is [H,T]  (one scalar per head and token)
ideal[d,h,t] = x[d,h,t] * (1/(1+exp(-gate[h,t])))
判据: x.ne[3]==1 && gate.ne[2]==gate.ne[3]==1 && gate.ne[0]==x.ne[1] && gate.ne[1]==x.ne[2]
```

**索引为什么是免费的一步除法**：x 的元素 `(d,h,t)` 在 `d + D*(h + H*t)`，gate 的元素 `(h,t)` 在 `h + H*t`；`D % 8 == 0` 时每个 8 元素 pack 完整落在一个 head 内 ⇒ `pack_index / (D/8)` **恰好**是 gate 下标（无取模、无 per-element gate 读）。标量内核用 `i / head_dim`，任意 head_dim 都对。

**代价**
- 额外权重：**0**（gate 仍是 `[16,2560]`，2.95 MB/模型）。
- 运行期：每 token 每层读写 x 各一次 = `2*(256*16*T*2)B`，T=1 时 16 KB ⇒ 36 层 576 KB/token，相对 8.23 GB 权重可忽略。
- 额外前置工作：**`g_proj` 的 linear 几何 `(16,2560)` 不能走 bf16 MMA**（`bf16_gemm_mma.cu:17` 的 `static_assert(kOutputRows % 64 == 0)`，16%64≠0）⇒ 两条出路：(i) 转换期把 g_proj padding 到 64 行（`64*2560*2 = 320 KB/层 → 11.5 MB/模型`）并在 `bf16_dispatch.cpp:21` 的白名单 + `bf16_gemm_mma.cu` 加一个 `Bf16GemvGeometry<64,2560>`（约 5 行）；(ii) **更省**：把 gate 拼进 `q_k_v_proj` 的同一个 GEMM（`[q(4096);k(1024);v(1024);gate(64)]` ⇒ n=6208，6208%64==0），一次 GEMM 出四路，只要一个 (6208,2560) 几何 —— 这条留给 S54-2 的对象计划决定。
- 家族运行时接线（约 12 行，属 S54-1/2 的目标骨架，不在本补丁内）：
  `text_context.h` 加第 8 个检测惯用法 `headwise_output_gate()`（回落 `false`，照抄 `:136-148` 的 `double_norm_attn_impl` 写法）；
  `attn_mix` 里 `Tensor gate = ModelConfig::headwise_output_gate() ? projection.gate.slice(0, 0, kCfg.n_q) : projection.gate.view({kCfg.head_dim, kCfg.n_q, T});`。
  `gate` 缓冲仍是 `query_size` 行（不动 recipe），Spark 的投影叶子只写前 `n_q` 行；`slice(0,0,n_q)` 给出 `{n_q,T}` 连续张量 —— **不需要手工构造 stride 的 Tensor**（这是选 `slice` 而不是 `view({1,n_q,T})` 的原因：`{1,n_q,T}` 的连续 stride 是 `h + n_q*t`，与缓冲的 `h*T + t` 不同）。

**验证**
1. op 级（新增用例，oracle 独立）：`run_headwise_case(256,16,1/6/12)`、`(128,32,3)`、`(256,4,5)` 五组 + `run_case([256,256])` 保证同形调用不被劫持。判据：pointwise `{2e-5, 4.05e-3}`（与既有 `sigmoid_mul` 同一 criterion），gate 输入逐字节不变，guard 区不被写。
2. **零影响证明**（可复跑，无需 GPU）：
   - `grep -rn 'ops::sigmoid_mul' src/` = 3 个调用点，全部 `[head_dim,n_q,T]`；
   - 新分支要求 `gate.ne[0] == x.ne[1]`（`n_q == head_dim`），三种几何 `16≠256 / 24≠256 / 32≠128` 都不满足 ⇒ 现有调用**结构上不可达**；
   - 既有 4 组同形用例（`[6144,1] [6144,48] [4096,17] [4096,128]`）逐字节不变。
3. 端到端：同 §3.2 的 logits/hidden 对拍 + "gate 置常数后 logits 必须变"。
4. 性能：目前不主张；若要，`bench` 侧加一个 exact-shape 点（`[256,16,1]` 与 `[256,16,512]`），对比 a/b 两条路线的 decode tok/s。

### 3.4 路线 (a) vs (b) 取舍

| 维度 | (a) 展开 g_proj | (b) headwise 路由 | 依据 |
|---|---|---|---|
| 引擎改动 | **0 行** | 6 文件 +153 行（+ 家族接线约 12 行，另计） | 本批 diff |
| 转换器改动 | 每层 4096 行复制 + artifact +755 MB | 无（原样 `[16,2560]`） | — |
| 额外权重/显存 | **+754.97 MB（+9.2%）** | 0（+2.95 MB 原始 g_proj，二者相同） | §3.2/§3.3 算术 |
| decode 吞吐 | −9.2%（权重带宽） | ~0（576 KB/token） | 权重 8.23 GB vs 755 MB |
| 额外 linear 几何 | `(4096,2560)`（**与 q 共用**） | `(64,2560)`（padding）或拼进 `q_k_v` 的 `(6208,2560)` | §3.3 |
| 数值等价性 | **结构性等价**（同一 BF16 权重、同一 GEMM，只是行重复；门控算术逐元素一致） | 等价（`x*sigmoid(g)` 与 HF 的"fp32 sigmoid→转 dtype→乘"差 ≤1 ULP，引擎只round一次） | HF `:200-206` |
| 新算子/新测试 | 无 | 1 条形状分支 + 1 个 op 级 oracle 用例 | — |
| 首版建议 | **v1 用 (a)**：先把"能出 token + 与 HF 对齐"做成，数值上不引入任何新内核 | 随后切 (b)：拿回 9% 吞吐与 755 MB | 与 S54 §7 的 v1 范围一致 |

> 我认为两者都不该删：**(a) 是收敛路径，(b) 是终态**。若只想留一个，留 (b)（把 755 MB 换成 6 文件 +153 行的代码量更划算），但那样 v1 的第一次数值对齐就会同时引入"新 op + 新几何"，排障面变大。

### 3.5 风险

| # | 风险 | 级别 | 处置 |
|---|---|---|---|
| B1 | 扩宽 `sigmoid_mul` 契约后，将来某个架构出现 `n_q == head_dim`（例如 128 头 × head_dim 128）且想用逐元素门 ⇒ 被判成 headwise | 低 | 契约里写明形状判据（含"两形态仅在单元素张量上重叠且结果相同"）；发生冲突时该架构改用逐元素形态需显式调用新符号（到时再定） |
| B2 | `pack_index / (D/8)` 依赖 `D % 8 == 0`；D 不是 8 的倍数时走标量内核（已实现），但没有 bf16x2 中间档 | 低 | 标量档已覆盖正确性；Spark D=256 走快路 |
| B3 | 家族接线（`headwise_output_gate()` + `attn_mix` 的三元判断）属 S54-1/2，本批未落 | 中 | 在 S54-1 骨架里一并落，且**必须与 27b/muse/35b 的回归同批**（检测惯用法回落 false ⇒ 理论零影响，但要跑一次 27b 冒烟证明） |
| B4 | 未做 GPU 验证（本任务约束） | — | §3.3 验证 1/2 是纯静态/op 级，落地后必须先绿 |

---

## 4. new_op #3：gated GELU MLP（`ops::gelu_mul`）

### 4.1 为什么必须要新 op（不是"低估工作量"的问题，是没有出路）

HF（`modeling_spark.py:126-132`）：`down(gelu(gate(x)) * up(x))`，`hidden_act="gelu"` ⇒ `ACT2FN["gelu"]` = **精确 erf**（与 `ops::gelu(GeluMode::Exact)` 同式，活体 `kernel/gelu.cuh:19-27` 用 `erff`）。
引擎现状：
- `ops::gelu`（`include/ninfer/ops/gelu.h:26`）是 **in-place 单元激活**（`void gelu(Tensor&, GeluMode, stream)`）⇒ 它能算 `gelu(gate)`，但算不出"再乘 up"。
- 全 op 清单里**没有两输入逐元素乘**：`include/ninfer/ops/` 下带 `*_mul` 的只有 `silu_mul`（`silu(g)*u`）与 `sigmoid_mul`（`sigmoid(g)*x`）；其余两输入算子只有 `residual_add`（加）与 `linear_add`。`logit_policy` 的 mult-only 是**标量**乘。
- `ops::linear_swiglu` 硬编码 SiLU（`silu_mul.h:9-21`、`linear_swiglu.h:37-53` 的公式与固定几何注册表），27b 用它、muse 拆成 `linear+linear+silu_mul`（`muse_glimmer_30b/impl/variant.cpp:365-379` 的 `post_mixer` 叶子）—— Spark 需要一个与之同形但激活不同的叶子。

⇒ 结论：**"让 MLP 前向按激活选择"不是可用路线**（家族里没有可选的两输入乘算子）；必须新增一个两输入逐元素 op。

### 4.2 补丁 C：`A3_spark_gelu_mul.diff`

按 `docs/maintainer/op-development.md` §4 的五件套（contract / wrapper / launcher / kernel / test）：

```
include/ninfer/ops/gelu_mul.h            (新, 28 行) 契约：ideal[i] = gelu(gate[i])*up[i]，精确 erf
src/ops/wrapper/gelu_mul.cpp             (新, 30 行) 校验 + 直接分派
src/ops/launcher/gelu_and_mul.h          (新, 16 行) 私有 launcher 声明
src/ops/launcher/gelu_and_mul.cu         (新, 54 行) bf16x8 / strided / 标量 三条路
src/ops/kernel/gelu_and_mul.cuh          (新, 89 行) 3 个 kernel，复用 gelu_one<false>
tests/ops/test_gelu_mul.cpp              (新, 164 行) oracle(FP64 erf) + 5 组用例
src/CMakeLists.txt                       (+2)        ninfer_ops 源清单两条
tests/CMakeLists.txt                     (+1)        ninfer_op_tests 列表加 gelu_mul
合计 8 文件（6 新增）+384/-0（427 diff 行）
```

要点：
- **激活与 `ops::gelu` 共用同一实现**：`#include "ops/kernel/gelu.cuh"` + `gelu_one<false>(...)`（精确 erf），不复制公式 ⇒ 两处不会漂移。
- **域只有精确 erf**：契约里写明"tanh 是另一个公式、需要自己的 criterion，不作为本 op 的模式"（§3"不冻结实现细节"，但公式必须唯一）。
- 不碰 `ops::gelu`、不碰 `ops::silu_mul`、不碰任何既有叶子。

### 4.3 代价

| 项 | 量 | 依据 |
|---|---|---|
| 代码 | +384 行（其中 164 是测试） | 本批 diff |
| 新 TU | 1 个（`gelu_and_mul.cu`，3 个 kernel，无模板爆炸） | 对比 i8 decode 档 ≈330 实例 |
| 运行期 | 每 token 每层读 2 写 1 = `3*10240*2*T` B ⇒ T=1 时 60 KB/层、**2.2 MB/token**（36 层） | MLP 中间维 10240 |
| 显存 | 0（复用 family 的 MLP 中间缓冲） | 由 Spark 的 `post_mixer` 叶子分配（同 muse `variant.cpp:368-374`） |
| 对既有模型 | **零影响**（见 4.4） | — |

### 4.4 对现有模型零影响的证明（三条，均可复跑）

1. **改动面**：补丁只改 2 个 CMakeLists（各加"新文件"一行）与 6 个新文件；`git diff --stat` 之后应当只出现这 8 个路径。
2. **无调用点**：`gelu_mul` 在落地后的调用点只有 Spark 自己的 `post_mixer` 叶子（新文件，S54-1/2 产出）；`grep -rn "gelu_mul" src/ include/ tests/` 在补丁应用后应只命中新文件 + tests 注册行。
3. **回归**：既有 op 测试三件（`ninfer_gelu_test` / `ninfer_silu_mul_test` / `ninfer_sigmoid_mul_test`）必须全绿且与改动前逐字节同结果（它们不经过新代码）。
4. （端到端）27b/muse 冒烟输出逐字同：MLP 叶子按 target 分文件，Spark 的新叶子不在它们的调用链上。

**风险**：C1 新增 TU 与 kernel 的编译成本（小）；C2 criterion 的选取 —— 起始用 `{2e-5, 4.05e-3}`（与 `silu_mul`/`sigmoid_mul` 同族；末次 bf16 rounding ≤ 2^-9 ≈ 1.95e-3 < 4.05e-3，绝对值项覆盖 `gelu(g)→0` 的下溢区），若首跑不达标，**先查 oracle 是否复现了 production 的 erf 近似差异**再谈放宽（§6.3：放宽需要数值理由）。

---

## 5. dry-run 原文（`patch -p1 --dry-run`，rc=0）

### 5.1 `A3_spark_head_geometry.diff`（8/8 OK）
```
checking file src/ops/kernel/gqa_attention_geometry.cuh
checking file src/ops/wrapper/gqa_attention.cpp
checking file src/ops/launcher/gqa_attention.h
checking file src/ops/launcher/gqa_attention_decode.cu
checking file src/ops/launcher/gqa_attention_decode_e8.cu
checking file src/ops/launcher/gqa_attention_prefill.cu
checking file src/ops/launcher/gqa_attention_prefill_e8.cu
checking file include/ninfer/ops/gqa_attention.h
RC=0
```

### 5.2 `A3_spark_headwise_gate.diff`（6/6 OK）
```
checking file include/ninfer/ops/sigmoid_mul.h
checking file src/ops/wrapper/sigmoid_mul.cpp
checking file src/ops/launcher/sigmoid_gate_mul.h
checking file src/ops/launcher/sigmoid_gate_mul.cu
checking file src/ops/kernel/sigmoid_gate_mul.cuh
checking file tests/ops/test_sigmoid_mul.cpp
RC=0
```

### 5.3 `A3_spark_gelu_mul.diff`（8/8 OK，含 6 个新文件的创建）
```
checking file src/CMakeLists.txt
checking file tests/CMakeLists.txt
checking file include/ninfer/ops/gelu_mul.h
checking file src/ops/launcher/gelu_and_mul.h
checking file src/ops/wrapper/gelu_mul.cpp
checking file src/ops/kernel/gelu_and_mul.cuh
checking file src/ops/launcher/gelu_and_mul.cu
checking file tests/ops/test_gelu_mul.cpp
RC=0
```

> 三个 dry-run 都跑在**影子树**（`_collab/a3_scratch/shadows/<name>/`，内容是活体文件的字节副本）上，命令形式与落地时相同：
> `cd <shadow> && patch -p1 --dry-run -i _collab/A3_spark_<name>.diff`。
> 没有对 `/home/user/ninfer-fusion` 执行过任何 patch（含 dry-run）。

---

## 6. 复现命令（全部可原样复跑）

### 6.1 重新生成三个补丁 + 重跑 dry-run
```bash
wsl.exe -e bash -lc "cd /home/user/ninfer-fusion && md5sum src/ops/kernel/gqa_attention_geometry.cuh ... > /mnt/c/Users/User/Documents/ziqinzhang/_collab/a3_scratch/live_md5.txt"
python C:/Users/User/Documents/ziqinzhang/_collab/A3_mkpatch.py
```
生成器只用 `_collab/a3_scratch/live/` 下的活体字节副本；锚点不唯一会 `ANCHOR FAIL` 退出（不会产出错补丁）。

### 6.2 活体树未被改动（本会话实证）
```bash
wsl.exe -e bash -lc "cd /home/user/ninfer-fusion && md5sum <36 个文件> > .../live_md5_after.txt; diff -q live_md5.txt live_md5_after.txt && echo TREE_UNCHANGED"
# 输出：TREE_UNCHANGED
```

### 6.3 镜像/活体漂移对照（S54 行号是否可用）
```bash
python C:/Users/User/Documents/ziqinzhang/_collab/a3_scratch/cmp_md5.py C:/Users/User/Documents/ziqinzhang/_collab/a3_scratch/live_md5.txt
# 期望：SAME=26 DRIFT=6 MIRROR_MISSING=1 LIVE_MISSING=5
```

### 6.4 group-4 调度数学的独立复核（不需要 GPU/编译器）
对 `TT ∈ 1..6` 手算 `RowTiles = ceil(TT*4/16)`、按 `gqa_attention_decode.cu:285-337`（i8）与 `:417-438`（nvfp4）取 `Wc`，验证 `(Wc/RowTiles)*PVNtPerWarp*8 == 256` 与 `PVNtPerWarp ∈ {2,4,8,16}`，即可复现 §2.1(c-3) 的表。

---

## 7. 汇总：文件 / 行数 / 是否落地

| 产物 | 文件 | 行数 | dry-run | 落地 |
|---|---|---|---|---|
| `_collab/A3_spark_head_geometry.diff` | 8 | +140/-21 | rc=0 | **未**（本批不落地） |
| `_collab/A3_spark_headwise_gate.diff` | 6 | +153/-5 | rc=0 | **未** |
| `_collab/A3_spark_gelu_mul.diff` | 8（6 新） | +384/-0 | rc=0 | **未** |
| 路线 (a)（转换期展开 g_proj） | 0（引擎）+ converter recipe（属 `tools/**`，本批不写） | 0 | — | 由 S54-2 落地 |
| 家族接线（`headwise_output_gate()` + `attn_mix` 视图） | 2（`text_context.h` / `text_context_impl.h`） | ~12 | — | 由 S54-1/2 落地，本批只给设计 |

**落地顺序建议**：`A3_spark_gelu_mul.diff`（独立、零风险）→ `A3_spark_head_geometry.diff`（与 C_s23 的 TU 拆分同窗，避免 ptxas 峰值叠加）→ `A3_spark_headwise_gate.diff`（与家族接线同批，配 27b 冒烟）→ 路线 (a) 作为转换器首版、路线 (b) 作为第二步切换。
