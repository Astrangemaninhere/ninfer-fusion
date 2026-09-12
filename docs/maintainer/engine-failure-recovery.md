# 引擎失败恢复设计（EngineCore 503 不可用态）

来源：`_TODO.md` §61b 附带发现 / §62 分工（状态 + 轻侧 agent 第 5 项，backlog 转正评估）。
日期：2026-09-09。状态：**设计（未实现）**；本文给出判定依据、改动面与验收门。

## 0. 问题（实测现象）
程序调用中抛异常后，**触发请求**返回真实错误，但**后续请求全部 503
`service_unavailable`** —— 引擎进入不可用态且无恢复路径（§61b 实测：req-1 抛错后
req-2 直接 service_unavailable，直到重启进程）。

## 1. 现状（代码实证）
- `EngineCore::submit` 在 `stopping_ || failed_` 时抛
  `RequestError(RequestErrorKind::Unavailable, "inference engine is unavailable")`
  （`src/runtime/engine/engine_core.h:181-184`、`:219-223`）；serve 映射为
  503 `service_unavailable`（`src/serve/generation_service.cpp:85-90`）。
- `failed_` **只有一处置位**：`fail_all_locked`（`engine_core.h:1782-1804`，
  `failed_ = true` 在 `:1786`），由 `worker_loop` 的兜底 `catch (...)` 调用
  （`:1892-1901`），随后 **worker 线程 `return`** —— 引擎整体停摆；全仓无 recover API。
- **关键事实**：`CUDA_CHECK` 失败直接 `std::abort()`
  （`src/core/device.cu:39-44`）→ worker 的 catch **只捕获宿主侧 C++ 异常**：
  `RequestError`（预期）、`std::logic_error`（不变量）、`std::invalid_argument`（校验）、
  `std::runtime_error` / `std::bad_alloc`（分配；`DeviceBuffer` 有 6×2s 重试后抛出）。
- `/health` 恒返回 `{"status":"ok"}`（`src/serve/http_server.cpp:364-366`）→
  503 期间健康检查仍报 ok，**不可观测**。

## 2. 缺口
1. **预期性错误与不变量破坏混为一类**：输入校验失败（RequestError / invalid_argument）
   与代码 bug（logic_error）都按「引擎中毒」处理 → 单请求错误放大为整机不可用。
2. **无恢复路径**：失败原因只出现在触发请求的错误里，后续请求只见 Unavailable；
   恢复只能重启进程。
3. **不可观测**：`/health` 不反映引擎状态，编排层无法区分「服务健康」与「引擎已死」。

## 3. 设计

### 3.1 异常分类（worker 边界唯一 catch）
| 类别 | 典型异常 | 处置 |
|---|---|---|
| 请求域 | `RequestError`（ContextLengthExceeded / ThinkingBudgetCapacityInsufficient / Cancelled / QueueTimeout / InvalidMedia） | **只失败该请求及其 lane**，引擎继续服务 |
| 资源域 | `std::bad_alloc`、分配类 `runtime_error`（重试耗尽） | 引擎停摆，但**可恢复**；记录原因 |
| 不变量 | `std::logic_error`（含 `invalid_argument` 用于内部不变量时） | 引擎停摆，**永久**（需 recover 或重启）；记录原因 |
| 其他 | 未知 | 同「不变量」 |

注：CUDA 错误走 `abort`，不进此表 —— 粘性错误语义下进程退出即正确处置。

### 3.2 lane 级失败（请求域）
- 新增 `fail_request_locked(request, error)`：`complete_error` + 释放该请求资源
  （现成原语：`resources_.release_failed_commit`（`engine_core.h:1077`）、
  `abandon_request`（`:154`）），**不碰其它 lane**。
- 归因：worker 的 catch 目前没有请求上下文 → 在 `run_prefill_step` /
  `run_decode_round` / `run_control_batch` 调用点记录当前操作主体
  （`failing_request_`），catch 按它归因；**无法归因时保守降级为引擎停摆**。
- 前提：请求域异常必须发生在「物理状态未半提交」的位置；若已进入 commit 中段，
  仍按停摆处理 → 由 3.4 从源头保证。

### 3.3 恢复路径
- 新增 `Engine::recover()`（公开 API）+ serve `POST /recover`（走现有全局鉴权）：
  ① `cudaGetLastError()` 清错 + 小分配探针确认上下文可用；
  ② 重建 Program（复用 `replan_target_kv` 路径：reset → create → 容量解析 → 图重捕获，
     见 `src/targets/registry.cpp:250-263`）；
  ③ 重启 worker 线程；④ 清 `failed_` 与原因。
- **仅显式调用**，不做自动重试（避免掩盖 bug；自动重试只对资源域可选加退避）。
- 与「图 PINNING」工作包联动：重建 = 图全量重捕获，后续由 pinning 降本。

### 3.4 源头治理（防「半提交」）
- **校验前移**：请求域校验（draft 宽度 / 上下文 / 预算 / 几何）在 `submit()` 或
  规划期抛 `RequestError`，**不在执行轮内抛**。§61b 的教训即此：宽度守卫在核内抛
  → 全机停摆。
- 执行轮内只允许「不可失败操作」+ 设备端错误（abort）。

### 3.5 可观测
- `/health` 增引擎状态字段：
  `engine: {state: "ready"|"failed"|"recovering", reason, since, last_request_error}`
  （failed 时返回 503 语义，编排层可据此摘流）。
- 失败日志：异常类型 + `what()` + 阶段（prefill/decode/control）+ 归因请求 id。
- 注意 `/health` 当前免鉴权（`http_server.cpp:305`）→ 详情字段需脱敏（只给类别，
  不给路径/张量名）。

## 4. 验收
1. **请求域回归门**：注入请求域异常（测试钩子）→ req-1 失败后 req-2 正常完成。
2. **不变量路径**：注入 `logic_error` → 引擎停摆 + `/health` 报 failed + reason；
   `POST /recover` 后服务恢复且无需重启进程。
3. **资源域**：模拟瞬时 OOM（占显存）→ 恢复路径可用。
4. **行为不变**：CUDA fault 仍 abort；无异常时性能零回退（对比 tok/s）。

## 5. 工作量与排序
- **P0**（0.5-1 天）：3.1 分类 + 3.2 lane 级失败 + 3.5 `/health`
  —— 改动集中在 `engine_core.h` 的 worker 循环与 `http_server.cpp`。
- **P1**：3.3 `recover()`（依赖 Program 重建路径，与图 PINNING 同源）。
- **P2**：3.4 逐目标审计执行轮内的 `throw`（校验前移清单）。

## 6. 风险
- lane 级失败若归因错误会留下脏状态 → 默认保守（无法归因即停摆），
  并以数值回归 + 57K 长测覆盖。
- `recover()` 重建 Program 会丢 KV 缓存/前缀缓存（与 §1 前缀缓存同源）→
  文档需明确 recover 是「最后手段」，正常路径仍是修 bug。
