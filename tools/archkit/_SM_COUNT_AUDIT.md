# SM-count 硬编码审计（W12 审计部分，只读）

- 日期：2026-09-09（本轮全部条目已逐条对照源码复核，file:line 均为真实检索结果）
- 背景：社区 fork `UDPSendToFailed/Azhu9701 ninfer-4090d` 用编译期宏 `NINFER_TARGET_SM_COUNT`
  （4090D=114 / 4090=128 / 3090=82 / 5090D=170）做设备适配；本审计评估把上游写死的
  SM 数换成运行时 `device.props.multiProcessorCount` 的可行性与改动面。
- 结论速览：**本仓没有 `NINFER_TARGET_SM_COUNT` 宏**（全仓 0 命中），SM 数全部以
  **字面常量**（170 / 85 / 42 / 340 / 680 / 510 / 1020）散布在 9 处代码点。
  其中 1 处（GDN gating 协作 kernel 驻留判定）是**正确性相关**，其余为性能调优。
- 构建上下文：`CMakeLists.txt:10` 默认 `CMAKE_CUDA_ARCHITECTURES=120a`（5090/Blackwell），
  上述常量全部按 170-SM 目标调出。

---

## 1. 方法（真实检索命令，可复现）

对 `src/ include/ tools/ apps/ bench/ tests/`（排除 third_party）执行：

```
grep -rn -iE "multiProcessorCount|sm_count|SM_COUNT|num_sms|NUM_SMS|numSMs|nSMs|smcount" --include=*.{c,cu,cuh,cpp,h,hpp,cc}
grep -rn "1728" ...            # 仅 eval 语料文本命中，代码 0 命中
grep -rn "NINFER_TARGET_SM" .  # 0 命中（fork 宏未进上游）
grep -rn -E "\b(170|85|340|680|510|1020|114|82)\b" ...
grep -rn -E "<<<[ ]*[0-9]+" --include=*.cu --include=*.cuh src   # 字面 grid 启动
grep -rn -iE "occupanc|cudaGetDeviceProperties|cudaDeviceGetAttribute|cudaOccupancy|props\." ...
grep -rn -iE "wave|persistent|resident_ctas|blocks_per_sm|ctas_per_sm|kRtx5090|SmCount" ...
grep -rn -iE "//.*(RTX 5090|170-SM|CTAs? per SM)" ...            # 注释线索
```

每条命中均人工回读了上下文（kernel 是否 grid-stride、是否 `__device__`、常量是否参与
scratch 容量）。行号以当前工作区文件为准。

## 2. 逐条发现表

| # | 位置 | 常量 | 用途 | 是否真 SM 耦合 |
|---|------|------|------|----------------|
| A1 | `src/ops/linear_attention/gated_delta_net/chunked/output.cu:9-11`（用例 :43-46） | `kRtx5090SmCount=170`、`kCtasPerSm=4`、`kTargetCtas=680` | `launch_output` 按"至多一个 5090 wave"计算 `jobs_per_block=ceil(logical_jobs/680)`，再切 grid | 是（纯性能；kernel 有 MULTI_JOB 模板，正确性不受影响） |
| A2 | `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:259-261`（launch :1192,:1197,:1231,:1237,:1246,:1252） | `kRtx5090SmCount=170`、`kPrefillBlocksPerSm=3`、`kPrefillPersistentBlocks=510` | Q4/QX prefill GEMM 的持久网格（grid-stride loop `for(work=blockIdx.x; work<total; work+=gridDim.x)`，:293），与 :264/:683 `__launch_bounds__(...,3)` 配套 | 是（纯性能；grid-stride 保证正确性） |
| A3 | `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp:130`（340）、`:138-139`（340/680） | `340 = 170×2`、`680 = 170×4` | `cooperative_27/35_grid_is_resident` 判定 SplitK>1 的协作 kernel 网格是否驻留；对应 `bf16_gdn_gating_proj_kernels.cu:280-290`（`cudaLaunchAttributeCooperative` + `cudaLaunchKernelEx`）与 `bf16_gdn_gating_proj_gemm_mma.cuh:296`（`cooperative_groups::this_grid().sync()`） | **是，正确性相关**：非 5090 上若 plan 仍按 340/680 放行，协作启动会因网格超驻留直接失败（CUDA_CHECK abort，`cudaErrorCooperativeLaunchTooLarge`） |
| A4 | `src/ops/kernel/gqa_attention_geometry.cuh:21` | `DecodeSplits = 85 × DecodeSplitScale`（85=170/2） | decode split-K 上限；host 侧 `gqa_attention_decode.cu:41,:59,:92`、`gqa_attention_decode_impl.cuh:47,:65,:98`；**device 侧** `kernel/gqa_attention_decode.cuh:81-93,:97-111`（:186-187 实际消费）；同时经 `gqa_small_t_launch_capacity`（`gqa_attention_decode.cu:99,:463-469,:484`）决定 partial acc/m/l scratch 容量 | 是（SM 派生；device 侧使用 + 容量语义，改造要分两步） |
| A5 | `src/ops/softmax_attention/dense/causal_cache/geometry.cuh:12` | `SmallTMaximumSplits = 85 × SmallTSplitScale` | causal small-T split 上限；host `small_t.cu:23,:41,:48`；device `small_t.cuh:93,:97-112,:123-127` | 同 A4 |
| A6 | `src/ops/kernel/bidirectional_gqa_attention.cuh:18` | `kBidirectionalGqaMaxSplit = 85` | split_capacity 校验上限（`launcher/bidirectional_gqa_attention.cu:104`）；当前 planner 的 split_limit≤64（`...cu:62-67,:74`），85 实际达不到 | 是（SM 派生守卫，低影响） |
| A7 | `src/ops/softmax_attention/common/context_query.cuh:18` | `kContextQueryMaxSplit = 85` | 同上（`dense/context/launch.cu:104`；split_limit 逻辑同构于 :62-74） | 同 A6 |
| A8 | `src/ops/launcher/rope.cu:17-18`（用例 :53） | `kLargeBlockWaveCapacity = 1020 = 170×6` | `tokens ≤ 1020` 时选 256 线程大 block（整网格一波驻留假设） | 是（SM 派生阈值，块尺寸启发） |
| A9 | host：`gqa_attention_decode.cu:73-80`、`gqa_attention_decode_impl.cuh:79-86`、`causal_cache/small_t.cu:66-68`；device：`kernel/gqa_attention_decode.cuh:109`、`causal_cache/small_t.cuh:110` | `kMax = 42 × SplitScale`（42≈170/4，注释"keep the 8K grid at or below one 170-SM wave after accounting for the geometry's KV-head count"） | INT8 T=6、window 5000-8198 档的 split 数上限 | 是（SM 派生 cap，性能；device 侧有副本） |

明确**查不到**的：`1728`（只在 `eval/corpora/perplexity-1m/data/zhwiki/01.txt` 等语料文本出现）、
`NINFER_TARGET_SM_COUNT`（本仓 0 命中）、fork 的 114/82/128/170 宏体系（上游无对应物；
`128` 在本仓全部是 tile/向量宽度/keys-per-split 档位，非 SM 数）。

## 3. 分类

### (a) 真·编译期 SM 常量（必须改运行时）
A1、A2、A3、A8 —— 直接以 170 或 170 的倍数出现在 host 侧 launch/规划代码。
A4、A5、A6、A7、A9 —— 以 85（=170/2）、42（≈170/4）出现；A4/A5/A9 含 device 侧副本和
scratch 容量语义，属于"要改但需分阶段"的 (a)。

### (b) 与 SM 数无关的固定常量（不要动）
- `src/ops/linear_add/w8/w8_linear_add_gemm_simt.cu:75-76`：`<<<2048/RowsPerCta,...>>>`，2048 是 kIntermediate（模型中间维），非 SM。
- `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu:28-31`：`kAdaptiveD3Blocks=5×kIntermediate(512)`、`kAdaptiveD4Blocks=5×(kHidden/4)`，工作量派生。
- `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:253-258`：kExpertBM/BN/BK=64、stages=2、warps=8 —— tile 常量。
- `src/ops/launcher/rope.cu:13-16`：kLargeBlock=256/kFullChunkBlock=192/kSmallBlock=128/kDefaultChunkTargetTokens=1024 —— 块尺寸/分块目标。
- `src/ops/linear_pair/w8/w8_pair_plan.cpp:48-49,:59`：路由表里的 673/680/784/1467/1680 是 token 列区间边界（680 与 170×4 纯数字巧合）。
- `src/ops/common/bf16_vector.cuh:22-25`：`kBf16x8CacheSizedMaxElements=32Mi`，L2/缓存 regime 阈值。
- 各 GEMM/GEMV schedule 的 `kMinBlocksPerSm`（`bf16_config.h:63`、`fp8_config.h:56,:83,:111`、`nvfp4_config.h:68,:101`、`w8_config.h:43`、`q4_small_t_mma.cuh:33` 等数十处）：`__launch_bounds__(threads, N)` 的编译器驻留提示，与 SM **数量**无关，不需要也不应随设备改。
- 单 CTA kernel：`launcher/scalar.cu`、`prefill_scan`（prefill_kernels :1167 附近）、`d2_warp`（decode :514 附近）、MoE router（:39 `__launch_bounds__(kRouterThreads,2)`）。
- GQA/causal 的 window 档位 4096/8198/16390 与 64/128/256/480 keys-per-split：token 语义档位。
- `bidirectional/context` planner 的 split_limit 32/38/40/48/64（上下文长度驱动）。
- `w8_linear_swiglu_decode.cu:16` kIntermediate=6144 等模型形状常量。

### (c) 疑似但需运行时实测
- `src/ops/sparse_moe/prefill/sparse_moe_prefill.h:20`：`kSparseMoePrefillWideMin=768`（wide-plan 档位，疑似按 510 个持久块调出）。验证：4090D vs 5090 用 `bench/ops/sparse_moe_bench.cu` 扫 tokens 700-900。
- `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu:266` 注释"Three path CTAs … for the 170-SM target"：Paths1/3/9 档位选择调优于 170 SM。验证：小 tokens decode 在不同 SM 数设备对比三档。
- `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:1150-1153`：adaptive 窗口 `tokens>=47..51`，疑似依赖 route-jobs 对 510 块的填充率。验证：同上 bench 扫 40-60 tokens。
- `src/ops/linear/nvfp4/nvfp4_config.h:189,:199-201,:249`：warp-count 交叉点，注释自述"RTX 5090 cold-cache winners / measured occupancy crossovers"。验证：nsight compute 对比寄存器/occupancy。
- GQA INT8 T=6 的 192 keys/split 档（与 42 cap 联调）。验证：gqa decode bench 按 window 扫。
- `src/ops/launcher/causal_conv1d.h:20` `kCausalConvParallelMaxTokens=32` 与 `wrapper/causal_conv1d_silu.cpp:313-315` 注释（"two CTAs per SM to T=24"）：测量驱动的 T 档位，无 SM 数字，换设备需重测。

## 4. 现有 device 上下文如何拿到 SM 数

- `src/core/device.h:13-34`：`struct DeviceContext { ... cudaDeviceProp props{}; ... }`；
  `src/core/device.cu:57` 构造时 `cudaGetDeviceProperties(&props, device_id)` 填充 →
  **`ctx.props.multiProcessorCount` 现成可用**。
- 注意：`DeviceContext::sm()`（device.cu:122）返回 `major*10+minor`（compute capability，
  如 120），**不是** SM 数，不能误用。
- 引擎侧（src/）目前 **0 处**使用 `multiProcessorCount`（唯一 props 消费是
  `src/targets/registry.cpp:110` 的设备名日志）。现成运行时范例在
  `tools/hbm_bandwidth_probe.cu:269-272,:317-320`：`cudaOccupancyMaxActiveBlocksPerMultiprocessor`
  × `prop.multiProcessorCount` 算 resident grid。bench/serve 侧（`bench/ops/linear_bench.cu:714`、
  `bench/ops/gated_delta_net_bench.cu:889`、`src/serve/request_log.cpp:1036-1043` 等）也各自
  `cudaGetDeviceProperties`，仅用于打印/记录。
- 障碍：ops 层 launch 链（`wrapper/sparse_moe.cpp:193`、`wrapper/rope.cpp:126`、
  `gated_delta_net/chunked/launch.cu:20,:42,:57,:71` 等）只穿透 `cudaStream_t`，不传
  `DeviceContext&`。纯 host 规划器（`bf16_gdn_gating_proj_plan.cpp`、gqa/causal split
  planner）更没有设备句柄。

## 5. 编译期 SM 常量清零清单 + 最小改动方案

清零目标（host 侧一次性可清）：A1（680）、A2（510）、A3（340/680）、A8（1020）。
分阶段目标：A4/A5/A9 的 85/42（涉及 device 副本与容量）、A6/A7 守卫（低优先级）。

1. **加统一入口**：`src/core/device.{h,cu}` 增加
   `int device_sm_count();`（`cudaGetDevice` + `cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev)`
   + function-local static 缓存；失败时回退 170 并告警，保证旧路径不炸）。
   这样 (a) 类全部调用点无需改函数签名（ops 层只有 stream 也能拿）。
2. **A2**：`sparse_moe_prefill_kernels.cu` 六处 launch 前
   `const int persistent_blocks = 3 * device_sm_count();` 替换 `kPrefillPersistentBlocks`；
   删除 :259-261 的 constexpr（`kPrefillBlocksPerSm=3` 保留作注释或与 `__launch_bounds__` 对齐）。
   guard：`persistent_blocks >= 1`；kernel 为 grid-stride，无需其他改动。
3. **A1**：`output.cu:46` 的 `kTargetCtas` 改为局部
   `const std::int64_t target_ctas = 4 * device_sm_count();`；后续 `jobs_per_block`/`grid_chunks`
   与 `v.check_grid` 原样保留。
4. **A3（唯一正确性项，建议优先）**：`bf16_gdn_gating_proj_plan.cpp:130,:138-139` 的 340/680
   改为运行时 `2×sm` / `4×sm`；**更稳**的做法是按 kernel 实测
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel, threads, smem) × multiProcessorCount`
   （"每 SM 2/4 个 CTA"本身是 5090+CUDA 13.1 编译产物的实测，见 :133-137 注释，跨架构不成立）。
   启动处（kernels.cu:280-290）已有 `CUDA_CHECK(cudaLaunchKernelEx(...))`，超驻留会以
   `cudaErrorCooperativeLaunchTooLarge` 显式失败——建议在 launch 前再加一次驻留复核 guard。
5. **A8**：`rope.cu:18` 的 1020 改为 `6 * device_sm_count()` 的运行时局部值（:53 判断用）。
   注：`6 CTAs/SM` 这一占用数本身也来自 5090 实测，跨设备可保守取 min(6, occupancy)。
6. **A4/A5/A9（二期）**：85 同时是 partial scratch 的容量上界，直接改小省内存、改大有越界风险。
   最小安全路线：**保留 85×scale 作为编译期容量上界**，仅把 host 侧 clamp（`gqa_attention_decode.cu:78`
   等）改为 `min(sm_count×scale/KVHeads, 85×scale)`；device 侧副本（`gqa_attention_decode.cuh:81-111`、
   `small_t.cuh` 同构函数）如需彻底清零，须把 sm 数作为 kernel 参数传入，改动面大，单独立项。
7. **A6/A7**：守卫 `> kXxxMaxSplit` 可顺手换成 `> device_sm_count()/2`，低优先级。

## 6. 与 fork 宏方案的对比结论

fork 用 `NINFER_TARGET_SM_COUNT` 仍是一设备一编译；上述最小方案用
`props.multiProcessorCount`/`cudaDeviceGetAttribute` 运行时取值，一个二进制覆盖
4090D(114)/4090(128)/3090(82)/5090(170)，且 (a) 类常量在更小 SM 数设备上只会退化性能、
不再触发 A3 的协作启动失败。A3 的每-SM CTA 数请用 occupancy API 实测而非沿用 2/4。
