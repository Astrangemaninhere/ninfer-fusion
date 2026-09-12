# 1Cat 剩余工作包设计（施工单）

来源：`_TODO.md` §43 三问盘点 / §62 分工（状态 + 轻侧 agent 第 4 项）。
日期：2026-09-09。本文是**设计与验收契约**；实现进度与实测数字记 `_TODO.md`。

范围：1Cat 借鉴清单里尚未落地的六项 —— 前缀缓存（含 GDN 循环态）、贪心 MTP 调度化、
图 PINNING、QPN NVFP4 中段 MT=2、per-arch 分发层、旧卡回归清单。
每节固定四段：**现状（代码实证）→ 缺口 → 设计 → 验收/风险**。

已有设计文档（本文不重复，只做引用与增量）：
`docs/maintainer/resource-scheduling-and-context-cache.md`（上下文缓存/checkpoint 全局设计）、
`docs/maintainer/replayssm-gdn.md`（GDN 规格性状态契约）、
`tools/archkit/_GPU_MATRIX.md`（档位矩阵与待移植项）。

---

## 1. 前缀缓存（含 GDN 循环态）

### 1.1 现状（代码实证）
- 复用决策已存在，是 **checkpoint/catalog 模型**而非哈希前缀表：
  `allow_prefix_reuse`（`include/ninfer/types.h:258`）、
  `PrefixReusePath` 六值枚举（`include/ninfer/types.h:622`：Root / PrivateEndpoint /
  PrivateTurnClosure / PrivateResponseReplay / PrivateLongAnchor / SharedStablePrefix）、
  决策体在 `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:471-561`
  （SharedStablePrefix 需 `CheckpointKind` + `prefix_matches`；MTP 复用门
  `:546-561` 要求 `source->mtp_kv_valid >= reuse_base-1`）。
- KV 页级 COW 已就绪：`logical_kv_store.h:130` `KVPrefixForkReservation`、
  `:183` `KVActiveSnapshotShape`、`:197-200` 注释「整页原地引用，只有非对齐可变尾页
  拿新物理页」。共享前缀对象：`program.h:478` `SharedPrefixState`（kv bundle +
  `StateImageHandle state` + identity + frontier + refcount）。
- GDN 循环态在**独立池**、随 StateImage 走：
  `src/core/linear_attention_state.h:14` `LinearAttentionStatePoolSpec`
  （conv_channels / conv_width / value_heads / slot_count）、`:25` 布局
  `{conv[], recurrent[]}`、`:64` 池对象（`copy_slot:83`、`zero_slot:84`）；
  载体 `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/state_image.h:26/125/154-162`
  （`copy_slot` / `copy_to_host` / `copy_from_host`）；
  槽生命周期 `state_image_store.h:144/152/166/353`
  （`reserve_destination` / `reserve_reset`(zero) / `activate_reset` / `begin_fork`）；
  尺寸推导 `layouts_impl.h:156-169`（`slot_count = max_concurrency + device_state_slots`）。
- 规格性 GDN 状态：`src/core/gdn_replay_records.h:11` `GdnReplayRecordSpec`
  （宽度 = draft_window+1）、`text_context.h:218` `GdnStateAction{UpdateInPlace,
  RecordForReplay}`、`include/ninfer/ops/gdn_replay.h:44` `GdnReplayFoldPlan`；
  契约细节见 `docs/maintainer/replayssm-gdn.md`（「一个已提交 checkpoint + 一段短
  原始转移日志」，不变量 §7:612-635）。

### 1.2 缺口（相对 1Cat `PREFIX_CACHE=1`）
1. **无内容寻址缓存表**：没有「token 序列 → 页」的哈希索引与 LRU 淘汰；只有有界
   catalog（`include/ninfer/types.h:110-124` `ContextCacheOptions`：
   `max_private_continuations` / `max_shared_prefixes` / `max_long_anchors_per_continuation`）。
2. **任意边界截断复用缺失**：现有复用点是 cataloged checkpoint / session 端点，
   1Cat 在任意 token 边界截断（Mamba-align 缓存）即可复用。
3. **GDN 态的部分回滚未落地**：fork 复制已就绪（`copy_slot`），但「从 checkpoint 的
   record 重放到截断点」还只是设计（`GdnReplayFoldPlan` 已定义，未接前缀缓存路径）。
4. **GDN speculative-state 优化未做**：`_GPU_MATRIX.md:51` 记「21 syncs/70 copies 消除」。

### 1.3 设计
- **两层命中**：在 `prefix_identity.h:40` `PrefixShortlistDigests`（现为候选短名单，
  注释 `:37-39` 明确「精确比较仍是权威」）之上加**跨请求滚动哈希表**：
  token 序列滚动 hash → 候选（checkpoint / fork 点）→ 仍走 exact identity 校验。
  淘汰用现有 `SharedPrefixState` refcount + LRU（按 checkpoint 粒度）。
- **任意边界截断**：复用 `logical_kv_store.h` 的 COW 语义 —— 对齐部分整页共享，
  非对齐尾页复制；截断点取 token 边界，无需页对齐。
- **GDN 态回滚**：截断点不在 checkpoint 时，从最近 checkpoint 的
  `GdnReplayRecord` 用 `GdnReplayFoldPlan` 重放到截断点（record 宽度已等于
  draft_window+1，重放成本 O(宽度) 而非 O(上下文)）。
- **换池交互**：`reload_kv` 会重建 Program（见 §3），缓存必须在换池时按 revision
  失效，避免复用旧池页。

### 1.4 验收 / 风险
- 验收：①同一 25K 文档二次请求（1Cat 基线场景 382 tok/s）记录复用命中率与首 token
  延迟；②任意边界截断（prefix N / N+37 / N+129）与全量 prefill 对拍 logits ≤1e-2；
  ③回归现有 prefix-reuse 测试 + 57K 针刺长测（§38 协议）。
- 风险：状态槽复制成本（52 层 × conv/recurrent D2D）可能吃掉短前缀收益 → 先做
  ≥1K token 的截断门限；catalog 命中率不足 → 需实测再定是否上全量哈希表。

---

## 2. 贪心 MTP 调度化

### 2.1 现状（代码实证）
- MTP 是**固定图宽 + 逐行 extent**：`program_impl.h:11965-11973`
  （`extent = min{mtp_draft_count, draft_window, max_by_budget, capacity-frontier-1}`，
  `target_valid_columns = extent+1`）；下一轮 extent 由设备端
  `src/ops/kernel/mtp_round.cuh:29-48` 按 budget/context/SVIP 计算，host 在
  `program_impl.h:9971-9976` 消费。
- 自适应策略**只用于 backend 选择 / DFlash2 降级**：`spec_decision.h:66`
  `choose_spec_backend`、`:81` `should_demote_dflash2`、`:101`
  `dflash2_acceptance_too_low`（`kSpecDemoteTokens=20480` 等）。
- 接受率统计**已记录但从不回馈宽度**：`SpeculativeStats.accepted_per_position`
  （`include/ninfer/types.h:602-611`），累加点 `program_impl.h:12046-12048`。

### 2.2 缺口
MTP 宽度没有 acceptance 反馈回路 —— 即 §43 的「贪心 MTP 调度化」（`_GPU_MATRIX.md:51`
估 +10-25 接受点）。

### 2.3 设计
- 在 `mtp_round.cuh` 的 `next_extents` 输入里增加**逐位置接受率**：维护每位置
  EWMA（device 侧或 host 侧均可，先做 host 侧最小改动）。
- 贪心策略：连续接受 → 加宽至上限；一次拒绝 → 收窄一档；**滞回**：连续 2 轮同向
  才变宽/变窄，防抖。
- 落点：`program_impl.h:12046` 更新每序列宽度目标 → 下一轮 ingress 写入（extent
  输入通道已存在，不需要新契约）。
- 注意：宽度变化可能触发 graph profile 切换/扩展（`draft_window` 是 family 常量），
  与 §3 联动评估。

### 2.4 验收 / 风险
- 验收：64K 长测（§35 协议）接受率对照 —— 基线 vs 自适应，目标 +10-25 接受点且
  tok/s 不降；qwen27 / 35B-a3b / Muse 三变体回归。
- 风险：宽度抖动导致图重捕获（用滞回 + §3 的 update 路径压成本）。

---

## 3. 图 PINNING

### 3.1 现状（代码实证）
- 图模型：`program.h:414-431` `DecodeGraphProfile/Topology/Family`；
  捕获/实例化 `src/core/decode_graph.h:10/36`、`decode_graph.cpp:54/105/124/143/149`。
- 形状变化走 **exec-graph update 而非重捕获**：`program_impl.h:685`
  `install_graph_profile` → `executable.update(profile.definition)`。
- 按需扩展**仅普通路径**：`graph_capture_ceiling`（`include/ninfer/types.h:164-168`）
  在 `program_impl.h:11030-11038` 过滤、`:11800-11810` 触发
  `extend_ordinary_graphs`、实现 `:11286-11335`；MTP/DFlash 路径恒全量捕获。
- 预算：`layouts_impl.h:872-912` `graph_allowance_bytes`（普通 12 MiB/batch 等）。
- 换池**必然全量重建**：`engine.cpp:524-539` `reload_kv_storage` →
  `registry.cpp:250-263`（`program.reset()` + 重建），因此全部图重捕获。
- 稳定性前提（架构文档）：`docs/maintainer/engine-architecture.md:452` ——
  「CUDA Graph 按合法 exact-B topology 建立，request identity 和 page IDs 是稳定
  输入数据，不是 graph key」。

### 3.2 缺口
1. 无任何「pinning / keep-alive」机制（全仓 grep `pinning|recapture` 无命中）。
2. MTP/DFlash 无按需扩展。
3. 换 KV 表/池 = 全量重捕获，`_GPU_MATRIX.md:50` 记该成本 0.7-2.6 ms/轮。

### 3.3 设计（两步）
- **步 A（换池不重建）**：把 `reload_kv` 的「析构 Program → 重建 → 重捕获」改为
  「换 KV 池引用 + 图参数重绑定」。依据：页 ID/表指针是图输入数据（上引架构注释），
  池对象地址不变时可只 update 节点参数；地址变化用 `cudaGraphExecUpdate`。
  最小先验路径：`use_cuda_graph=false` 快捷路径先验证换池语义，再打开图。
- **步 B（按需扩展推广）**：把 `extend_ordinary_graphs` 的机制推广到 MTP/DFlash
  family（现在只有 ordinary 有）。
- 与 §1 联动：换池 = 缓存 revision 失效点。

### 3.4 验收 / 风险
- 验收：①reload_kv 前后 decode 首 token 延迟差 <5%；②换池不再触发全量重捕获
  （日志计数）；③三变体数值回归（换池后 logits 与换池前同表一致）。
- 风险：页表/workspace 指针重绑定错误会**静默错算** → 必须配 CUDA graph 参数审计
  与数值对拍（不可只测「不崩」）。

---

## 4. QPN NVFP4 中段 MT=2

### 4.1 现状（代码实证）
- **内核已就绪且已验**：`src/ops/linear/qpn/qpn_kernels.cuh:876-903`
  （`skinny_nvfp4_qpn<MT>`，MT=2 语义见 `:881-882`、`:1311-1325` 实测 298 vs 431 GB/s）；
  FP8 兄弟 `skinny_fp8_qpn8_mt2`（`:1326-1327`）；QPN2 M≤8 `:1066-1077`；
  e2e 测试 `qpn_mma_e2e_test.cu`（M=16，commit `2bcb4f3` PASS）；槽映射
  `qpn_map.cuh:10-29`（qpn2 与 simt 仅 M≤8 行映射不同）；prepack 逐字节一致
  （`tools/archkit/qpn_port/qpn_prepack_proto.py`）。
- **缺 host 入口与引擎接线**：`qpn_host.cu:13-15` 只有 `gemm_qpn_simt`（M1-3，
  `m>3` 直接抛）；`src/ops/linear/qpn/*` 未进任何 CMakeLists；`linear.cpp` 的
  qtype 分派（`:78-107`）无 QPN 档。

### 4.2 缺口
host 按 M 选 MT 的入口、权重 prepack 旁路、CMake 接线、M9-16 标定、真机 V100。

### 4.3 设计
1. **host 入口 `gemm_qpn`**：M 分档 —— simt M≤3 / `qpn<1>` M4-8 / `qpn<2>` M9-16 /
   wmma ≥17；沿用现有模板实例化。
2. **prepack 旁路**：转换期或加载期做 QPN2-W4A16 prepack（原型已证逐字节一致），
   artifact 加 layout 标记或加载期重排；复用 `layouts.py` 的 layout 注册表。
3. **CMake 接线**：qpn 目录进构建 + `__CUDA_ARCH__` 守卫共存（与 §5 一起做）。
4. **标定**：本机 sm_120 用 `compute_70` PTX-only JIT 做数值验证
   （`tools/archkit/qpn_port/qpn8_test_sm120.py` 已备）；真机 V100 待 CUDA12 链。

### 4.4 验收 / 风险
- 验收：M9-16 与 FP8/参考实现对拍 ≤1e-2（JIT 或真机）；接入后 qwen27 NVFP4 档
  数值回归不变。
- 风险：本机无 sm_70 真机 → 性能数据缺位，**如实标注**，不得用 JIT 数字冒充真机。

---

## 5. per-arch 分发层

### 5.1 现状（代码实证）
- 构建门：`CMakeLists.txt:9-23`（≥sm_75，nvfp4 TMA 仅 sm_100a/120a 编译期守卫）。
- 运行时**硬拒**：`layouts_impl.h:824-835`（`device.sm() != 120` 直接 throw）。
- `sm()` 全仓仅 3 处调用：`src/core/device.h:31` / `device.cu:122`（定义）/
  `registry.cpp:110`（日志）+ 上述门。
- 内核选择全部**按 shape/dtype**：`linear.cpp:78-107`（qtype 分派）、
  `nvfp4_dispatch.cpp:20-48`（`resolve_route` A16/W4A4）、gqa launcher 的几何 if/else。
- `__CUDA_ARCH__` 守卫几乎为零（仅 `iso_codec.h`、gdn `common.cuh`）。
- 目标路由表（设计意图）：`_GPU_MATRIX.md:8-15`
  （100/120a→nvfp4 W4A4；90/89→fp8 / W8A16；86/80/75→groupwise-int；70→QPN2 W4A16）。

### 5.2 缺口
无中心 per-arch 表；现有唯一 arch 行为是「非 sm_120 拒绝」，与「非门禁、按卡走内核」
的既定路线相反。

### 5.3 设计
- 引入 `KernelRoute` 枚举 + 中心选择函数：
  `route = select_route(device.sm(), profile, problem_shape)`；
  linear 包装层按 route 选实现（现有 `resolve_route` 扩展为跨格式，而不是只分
  nvfp4 内部的 A16/W4A4）。
- **非门禁原则**：未知 sm 不再 throw，退回保守路线（SIMT / groupwise-int）并 warn；
  只有真正不支持的组合才给指引文案（沿用 `layouts_impl.h:829-833` 的措辞）。
- 内核文件按 `__CUDA_ARCH__` 守卫共存，CMake 多 arch 编译，运行时查表。

### 5.4 验收 / 风险
- 验收：sm_120 上 route 选择日志 + qwen27/35B/Muse 数值回归不变；旧档用 JIT 验证。
- 风险：无老卡真机 → 表只能「纸面 + JIT」，与 §6 的回归清单配套补齐。

---

## 6. 旧卡回归清单

### 6.1 现状
- **仓内无老卡 harness、无基线数据**（全仓 grep 老卡名只命中文档与 GUI 分类）；
  现有 bench 全 sm_120 目标（`bench/README.md`、`tools/bench/*`），无 CI。
- 散文基线：`_GPU_MATRIX.md:36`（3090/4090 groupwise-int 195-203 tok/s）、
  `:59`（4×V100 chain-MTP 366 tok/s）、`RESEARCH-EXTERNAL.md:7-8`。

### 6.2 设计（清单化）
- 档位与路线：sm_70（V100）QPN2-W4A16；sm_75/80/86（2080Ti/A100/3090）
  groupwise-int；sm_89（4090）fp8；sm_90（H100）fp8/W8A16。
- 每档最小回归集：
  ① **数值对拍**（该档 GPU vs CPU 参考，沿用 iso_kv/suffix/split 的对拍模式）；
  ② **端到端冒烟**（单模型 512 输出，记录接受率与 tok/s）；
  ③ **量级门**（对照社区基线 195-203 / 366 tok/s，只作 sanity，不作硬门）。
- 机器策略：本机无老卡 → ① CUDA12 链 + PTX JIT 先跑通数值；
  ② 用户/社区真机数据回填；③ 拿到 V100 实机再跑全清单。
- 产物：`tools/bench/old_gpu_baseline.json`（schema：gpu / sm / profile / model /
  ctx / tok_s / accept / date / source[real|jit|community]）+ 对照脚本 `--compare`。

### 6.3 验收 / 风险
- 验收：每档至少一条数值对拍 PASS + 一条 tok/s 记录（来源必须标注）。
- 风险：无真机 → 性能项长期空缺，清单**如实标注 pending hardware**，不造数。

---

## 7. 建议实施序（价值 × 依赖）

| 序 | 工作包 | 理由 | 前置依赖 |
|---|---|---|---|
| 1 | §2 贪心 MTP | 纯调度改动、内核零改动，接受率收益直接 | 与 §3 宽度-图交互 |
| 2 | §4 QPN MT=2 接线 | 内核/e2e 已就绪，只差 host 入口 + CMake，解锁 V100 档 | §5 CMake 共存 |
| 3 | §3 图 PINNING 步 A | 消除换池全量重捕获，与 reload_kv 主线同源 | 数值对拍基建 |
| 4 | §1 前缀缓存内容寻址 + 任意边界截断 | 工程量最大、收益场景明确（RAG/agentic 复述） | §3 步 A 稳定 |
| 5 | §5 per-arch 表 | 依赖 §4 接线与 §6 验证手段 | §4 / §6 |
| 6 | §6 旧卡回归清单 | 与 §5 同步推进，先立数据 schema | CUDA12 链 |

注：本文档为设计与验收契约；每项落地后在 `_TODO.md` 追加实测结论（含失败数据）。
