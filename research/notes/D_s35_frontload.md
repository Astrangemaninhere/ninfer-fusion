# S35 (agent D): §3.4 源头治理 — 执行路径 throw 审计 + 校验前移补丁

日期 2026-09-10 · 基线树 `ninfer-fusion-repo @ 0eaac04`（4 个目标文件 pristine，未动 src/）
产物：`_collab/D_s35_frontload.diff`（4 文件/161 行，anchored，`patch -p1`-able，**只评审未落树**）
生成器：`_collab/D_s35_mkpatch.py` · 验收门：`_collab/D_s35_gate.py`
设计来源：`docs/maintainer/engine-failure-recovery.md` §3.4（P2）。S31（P0 残差）补丁保持未应用未改动；两者将在窗口 H 一起批（`_window_h.sh` 清单）。

## 0. 方法与范围

从三个执行体派发点（`run_prefill_step` / `run_decode_round` / `run_control_batch`，engine_core.h:1376/1790/1803）加 worker 边界（`worker_loop` try 块 :2019-2123，含 admission）做**调用图闭包**，逐层枚举 throw：
L1 引擎宿主（engine_core.h 执行/边界/admission 函数）；L2 引擎内部层（scheduler.h / resource_manager.h / request_record.h / generation_budget.h）；L3 目标执行层（program_impl.h：advance_prefill:7194 → advance_prefill raw:11459、decode:8585 → decode_raw:12275 → ordinary/mtp/dflash/dflash2 批、append_forced_tokens:8621、commit:8818、abort_pending:8946、wrap_*:6916/6941、validate_licensed_tokens:11451）；L4 输出策略（frontend.cpp preview_model:1030 / preview_control:1145 / preview_terminal:1202 / commit_preview:1219 / validate_generation_capacity:1185）；L5 规划器（request_plan_impl.h plan_request:217 + validate_sampling:16 + inspect_lane:~456 + select_shared_captures:1246）。
目标家族共享一个 runtime（`request_plan_impl.h` 仅 qwen3_6 一份；muse/qwen27b/35b/qwen4_exp 经 Variant/TextConfig 复用）→ 一处审计覆盖全部目标。
判定依据（"哪个参数携带请求依赖"）逐条标注在下表。

## 1. 审计表（全部 throw 位点，按类）

### 类 (a) 请求域、可前移 —— 本补丁全部转换（10 处，均在规划期，消息逐字不变）
| 位点 (request_plan_impl.h) | 条件 | 请求依赖来源 | 转换后 kind |
|---|---|---|---|
| :21/:24/:27 validate_sampling | temperature/top_p/min_p 非法 | `ResolvedExecutionOptions.sampling` ← HTTP 请求体；serve 层**无任何**先行校验（grep generation_service.cpp 零命中）——这是采样参数唯一校验点 | InvalidPrompt |
| :219 | prompt 空 | 公开 API `Frontend::prepare_tokens` 可传任意 token 序列；serve 文本路径有 frontend.cpp:338 兜底 | InvalidPrompt |
| :221 | prompt 超 capacity | 用户 prompt 长度；同一条件在 frontend prepare 期已用 `ContextLengthExceeded`（frontend.cpp:244-246）→ kind 对齐 | ContextLengthExceeded |
| :228 | token 超词表域 | 同 :219（prepare_tokens 路径用户可控；serve 路径为内部兜底） | InvalidPrompt |
| :247 | Vision 关闭却带媒体 | 用户媒体输入 vs 引擎配置；serve prepare 期同条件已 400 "vision_disabled"（generation_service.cpp:324/:379） | InvalidMedia |
| :314/:323 | rewrite checkpoint/frontiers 非法 | `prompt.identity.rewrite_*` ← 用户 PromptInput cache marker（frontend.cpp:709-750 产生） | InvalidPrompt |
| :338 | capture opportunity frontier 非法 | 同上（context_cache.opportunities ← 用户 cache marker） | InvalidPrompt |

附带 (a) 级**结构缺口**（engine_core.h）：`try_admit_one` 里 `ensure_base_plan`（规划）已按请求隔离（:1677-1683/:1751-1757 catch → `remove_pending_error`），但 **`inspect_admission` 两处调用点（:1684/:1758）不在 try 内** —— 规划期异常直接落到 worker 边界、`unit_owner_` 为空 → fail_all → 整机停摆。本补丁把两处 inspect 折进同一 per-request try（与 ensure_base_plan 同款模式）。`admit_planned_request` 深处的 `reserve_materialization` **不折**：它开启 context transaction 且已改 scheduler 状态，异常时保守停摆是 §3.2 的既定语义（见 §2 不可前移清单）。

### 类 (b) 不变量/内部契约 —— 保持停摆语义，不动（抽样列举，全量已过 grep）
- engine_core.h 执行体：run_prefill_step:1380-1388、resolve_prefill_progress:1350-1362、commit_pending:1072/1105/1112/1133/1150/1190/1198/1202/1252（ragged 布局/lane 绑定/容量预留/commit 行对齐——全是引擎自有状态契约）；run_control_batch:1811/1830/1842/1851/1855。
- program_impl.h 执行层：advance_prefill:7196-7201、decode:8590-8601（membership 由 scheduler 从引擎状态构造，非用户数据）、decode_*/append_forced_tokens 的 frontier/ledger 守卫（:12306-12333、:8643-8654——引擎状态 + budget，budget 由引擎派发）、inspect_admission:1303-1322、select_shared_captures 的 RM↔Program 契约（request_plan_impl.h:469/:1254/:1263）。
- frontend.cpp 预览路径：preview_model:1033-1047/1073、preview_control:1147-1176、preview_terminal:1203-1208 —— 仅 logic_error/invalid_argument（license 一致性、引擎双花预算），**零 RequestError**。
- request_plan_impl.h 内部兜底 invalid_argument ×8（保留，gate 按消息白名单锁定）：:233 metadata shape、:237/:243 media payload shape、:298/:304 vision span/envelope（`plan_vision_control` 自算，用户不可直达）、:469/:1254/:1263 RM/Program 调用契约。

### 类 (c) 设备/资源
- `DeviceArena::alloc_bytes`（arena.cu:214-265）：backing 缺失 runtime_error / 非零对齐 invalid_argument / **bytes==0 invalid_argument（:236，带 dladdr+backtrace 取证，今日事故位点）**/ overflow_error / **耗尽 bad_alloc（:265）** → §3.1 已分类 resource/invariant，S31 补丁把 cudaMalloc 前缀补进 resource。执行路径上的 `work.alloc`（program_impl.h:1179-1181/8666 等）尺寸全部来自启动期冻结的 workspace 计划或带守卫的引擎量（row_stride≠0、checked_i32）——**当前树上没有请求几何直达执行期 arena 尺寸的通路**（§61b 的 TT 守卫已撤、alloc-zero 已有取证+分类兜底）。
- `CUDA_CHECK` → abort（device.cu:39-44，未触及）。
- 计划期异常的既有隔离（`ensure_base_plan` catch-all）对 invariant 也按请求失败——预存在的宽仁，本补丁不改其语义，只把 inspect 纳入同一模式。

## 2. 不可前移项（输入只在执行期存在，逐项点名输入）
| 位点 | 为什么不能移 | 执行期输入 |
|---|---|---|
| frontend.cpp preview_model/preview_control | 决策消费的 token 是设备产出的 | 采样 token ids（decode 批）、canonical thinking-control span |
| program_impl.h validate_licensed_tokens:11451 | 校验的是设备/前端产出的控制 token | `pending_control_tokens()`（前端生成的 canonical 控制序列） |
| commit_pending 的 preview/commit 链 | Commit 相位（lane 恢复不可用），但全部 invariant 类 | row_tokens、budget remaining |
| append_forced_tokens:8666 checked_i32 | 上界由引擎选的 row_stride 决定且非零守卫在前 | membership.row_stride（引擎量） |

## 3. 补丁内容与影响面
4 文件：`include/ninfer/types.h`（枚举 +InvalidPrompt）、`src/serve/generation_service.cpp`（switch +1 case，穷尽 switch 无 default → 编译器强制补映射）、`src/targets/qwen3_6/impl/runtime/request_plan_impl.h`（+include ninfer/types.h，10 处 invalid_argument→RequestError）、`src/runtime/engine/engine_core.h`（admission 两处折 try，+10 行）。
调用点数：10 个 throw 转换 + 2 个 try 块扩展；**happy path 零行为变化**——所有转换只在失败路径改变异常类型；检查条件、消息字节不变；`RequestError` 派生自 `invalid_argument`，现有按 `invalid_argument` 的 catch（如 engine submit :206-211 的模式）不受影响。
外部身份变化（有意）：规划期请求域错误从 500 internal_error（`wait()` 只 catch RequestError，generation_service.cpp:424，存储的 invalid_argument 直落 HTTP 兜底 500）→ 400 `invalid_prompt`/`context_length_exceeded`/`invalid_media`；`invalid_prompt` 与 prepare 路径既有 code 字符串（:369/:404）一致，日志 grep 兼容。

## 4. 验收
**现在可跑（静态门，CI 形态，可反复执行）**：
```
wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/D_s35_mkpatch.py     # → S35_PATCH_DRYRUN_OK（逐文件+combined 4/4+1 全 OK）
wsl.exe -e python3 /mnt/c/Users/User/Documents/ziqinzhang/_collab/D_s35_gate.py <root> # 门：G1 执行体仅注入钩子可抛 RequestError；G2 program_impl.h 零 RequestError；G3 预览函数零 RequestError；G4 引擎内部层零 RequestError；G5 规划器 ≥10 RequestError 且剩余 invalid_argument 逐消息=8 个审计兜底；G6 InvalidPrompt 存在且被 serve 映射
```
实测（2026-09-10）：shadow 树（/tmp/s35shadow，S31+S35 已应用）→ **S35_GATE_PASS**；pristine 树负对照 → `GATE-FAIL: ... only 0 throw RequestError (want >= 10)`（门确实能抓未修状态）。叠加序两向验证：S31→S35 与 S35→S31 均 offset 应用无 fuzz 无 reject。
**只有 GPU/重建窗口能证明**：① 编译（request_plan_impl.h 新引用 ninfer/types.h；穷尽 switch；`ResourceInspection` 默认构造——成员有 NSDMI，已核实 resource_manager.h:237-240）；② e2e：`top_p=7` 请求 → 400 `invalid_prompt` 且引擎存活（下一个请求 200、`/health` ready）；③ S31 四门 + 本门一起回归；④ happy-path tok/s 零回退（补丁不进热路径）。
窗口 H 批次提示：与 S31 同涉 `include/ninfer/types.h`（公共头），两 patch 叠加已验证，建议同批一次编译。

## 5. 与设计的偏差声明
- §3.4 文案"在 submit() 或规划期抛 RequestError"——规划期是允许位置，本补丁把规划期的**类型**补正为 RequestError 并把 admission 规划段整体请求域化，未把检查上移到 submit()（上移需把 EngineOptions/capacity 传给 frontend 层，收益低于风险；规划期每请求失败语义已足够）。
- `ensure_base_plan`/新增 inspect 折叠沿用既有 catch-all（invariant 也按请求失败）。这是落地代码的既有选择；严格 §3.1 会要求 invariant 停摆。保留现状的理由：规划是纯读路径（RM::inspect 只读、候选只是计划），异常无半提交状态；若未来要收紧，gate 的 G5 白名单是加分类的挂点。
