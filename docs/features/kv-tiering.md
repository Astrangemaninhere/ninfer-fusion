# KV 温冷分层、冷池与卸载

「温冷分层」在这棵树里分成三层实现：**决策层**（谁该冷）、**介质层**（冷层放哪）、**衔接层**
（页与池怎么接）。三层的入口与限制分别写在下文，每条带出处。

## 1. 冷池策略与保留量（CLI 面）

| 开关 | 语义 | 出处 |
|---|---|---|
| `--cold-policy none\|off\|window\|host\|disk\|host-then-disk` | 冷池策略；`host+disk` 是等价拼写 | `apps/cli/options.cpp:612`（`host+disk` 在 `:621`）；默认 `None`（`apps/cli/options.h:111`） |
| 给策略时的副作用 | **只在未显式给 keep-tokens 时**把 `cold_keep_tokens` 设 128 | `apps/cli/options.cpp:618-620`（自述这条修过「无条件覆盖」的缺陷） |
| `--cold-keep-tokens N` | 冷池保留 token 数 | `:628`，默认 **128**（`apps/cli/options.h:112`） |
| `--cold-host-bytes N[g\|m\|k]` | 钉住的 host 冷层预算 | `:631`，默认 **7 GiB**（`apps/cli/options.h:116`）；`apps/cli/options.h:114-115` 自述为什么不是 4 GiB：1M 带需要 `F - D = 5,634` 页 = 6.2004 GiB @ 1,181,745 B/页，而 `4ULL << 30` 会让 `cold_host_window_band` 为空 |
| `--max-cold-pages N` | 冷池显式页上限，**0 = 由策略推** | `:633`，默认 0（`apps/cli/options.h:121`）；`apps/cli/options.h:117-120` 自述（原文英文，此处转述）：不给的话 CLI 只能用推导出的 18 页，
`cold_keep_tokens/kPagedKVPageSize + 16`，于是「给卸载设上限」这件事在这个前端不可达 |
| `--cold-disk-path DIR` / `--cold-disk-bytes N` | 冷层落盘目录与预算 | `:650` / `:652`，默认 32 GiB（`apps/cli/options.h:130`），**0 拒绝**（`:654-656`） |

## 2. 介质层：host 冷层（`--cold-policy host` 的消费者）

`src/targets/qwen3_6/impl/runtime/cold_host_tier.h`，逐页一个**钉住的 host extent**，由
`--cold-host-bytes` 界定。关键函数与行号：

| 函数 / 类型 | 作用 | 行号 |
|---|---|---|
| `cold_host_page_is_read_free` | 页「读免费」判定 | `:81` |
| `cold_host_page_is_fetchable` | 页「可取」判定 | `:121` |
| `cold_host_tier_can_admit` | 准入判定 | `:155` |
| `struct ColdHostWindowBand` | 窗口带宽 | `:263` |
| `cold_host_window_band()` | 算窗口 | `:289` |
| `cold_host_window_refusal()` | 窗口关不上缺口时的**具名拒绝** | `:310`（拒绝语起点 `:310-343`） |
| `cold_host_bytes_for_window()` | 反推需要多少 host 字节 | `:385` |
| `struct ColdHostWindowRoundCost` | 逐轮代价 | `:420` |

两处自述值得原文照引：

- 该头文件明说 `--cold-policy host` 与任何 `--cold-host-bytes` 在结构上**是惰性的**
  （`cold_host_tier.h:94-97`），并给出 4 GiB → 3,637 页的算例（`:181`）。
- 窗口关不上缺口时，报告「按至少 N 页提高 `--cold-host-bytes`」，且带宽为空时返回空串
  （`:310-343`）；`:331-332` 自述这段**就是树自己对「窗口何时关不上缺口」的陈述**。

## 3. 决策层：hot/warm/cold 的闭环保留策略

`src/serve/kv_cold_policy.h`，**纯决策层**（host-only，只用 `std`）。`src/serve/kv_cold_policy.h:2-4`
自述它属于「S34（N4 / 用户 Q4）：闭环 hot/warm/cold 驻留策略 —— FreeToken 观测抽头上的纯决策层」。

| 类型 / 函数 | 作用 | 行号 |
|---|---|---|
| `struct ColdObservation` | 一轮观测 | `:52` |
| `struct ColdPolicyConfig` | 配置；`demote_quantile = 0.25`、`stable_cycles = 2` | `:58`、`:62`、`:63` |
| `struct ColdPolicyState` | 跨轮状态（EWMA 与 streak） | `:71` |
| `struct ColdDecision` | 决策结果 | `:78` |
| `cold_cut()` | 冷线：非深层、已观测层上 EWMA 的 `demote_quantile` 分位 | `:106`（分位读取 `:116`） |
| `decide_cold_residency()` | 主决策 | `:124`（`cut` 计算 `:157`） |

规则（都在这一个文件内，带行号）：

- 降级需要 EWMA 在**连续 `stable_cycles` 个窗口**都 `<= cold_cut`：`:204`、`:208`；驻留期要求
  写在 `:166`。
- **未观测层永不被降级**：`:224` 起（`above_streak` 的追踪）。
- 置信不足的窗口**不能把冷层提升出池**（否则一空闲就弹出）：`:313-316`。
- dry 模式只打印 `[ft][cold:dry] would ...`，不动任何东西：文件头 `:22`、`:35`。

**限制（必须一起写）**：这一层由服务端的 `--kv-auto-relayout` 周期驱动（`src/serve/serve_options.cpp:535`）；
**CLI 前端没有决策周期**，所以 `apps/cli/options.cpp:292` 明说它的消费者是 serve 侧特性。

## 4. 冷层 census 与页面单位自检

| 功能 | 是什么 | 出处 |
|---|---|---|
| 冷**回退** census | 被检查率 / 截断的统计 | `src/targets/qwen3_6/impl/runtime/cold_fallback_census.h:58,59`（宏名 `NINFER_COLD_FALLBACK_CENSUS_INSPECTED_RATE`、`..._TRUNCATION`）；同目录另有 `.sha16.txt` 指纹文件把它单独盯住 |
| 冷**重取** census | host 冷页被重新取回的统计 | `src/targets/qwen3_6/impl/runtime/cold_refetch_census.h:82`（`NINFER_COLD_HOST_REFETCH_CENSUS`） |
| 冷页单位自检 | 冷页单位的自检头 | `src/targets/qwen3_6/impl/runtime/cold_page_unit_check.h`；测试 `tests/test_cold_page_units.cpp`（另有 `.pre_land` 前像）、`tests/test_cold_slot_release_bytes.cpp` |
| 冷层代价模型 | 冷池容量/代价的独立预算头 | `src/product/kv_cold_tier_budget.h`（255 行），被 `cold_host_tier.h:48` include |

## 5. 衔接层：策略 → 分页介质，以及页池

| 功能 | 是什么 | 出处 |
|---|---|---|
| 策略到介质的映射 | `PagingColdMedium{None, DeviceWindow, PinnedHost, Disk, HostThenDisk}`，五态 switch **每个臂都在** | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2104-2108` |
| 分页 KV 池一族 | `PagedKVLayerView` `:35`、`PagedKVBatchLayerView` `:65`、`KVPlaneGeometry` `:91`、`PagedKVPlaneOrder` `:100`、`KVPageGeometry` `:105`、`DeviceKVPagePoolSpec` `:113`、`KVExecutionTableSpec` `:118`、`DeviceKVPlaneLayout` `:123`、`DeviceKVPagePoolLayout` `:128`、`DeviceKVPageHandle` `:154`、`DeviceKVPageLease` `:179`、`DeviceKVPageReservation` `:208`、`DeviceKVPagePool` `:236` | `src/core/paged_kv_cache.h`（行号为实测 grep） |
| 页池预分配 | 见 `kv-compression.md` 第 5 节 | `layouts_impl.h:2093` |
| host KV arena / extent store | host 侧 KV 大块内存与冷层 extent 存储 | `src/core/host_kv_arena.h` / `.cpp`；`src/targets/qwen3_6/impl/runtime/host_kv_extent_store.h` |
| 逻辑 KV 视图 | 逻辑 KV store | `src/targets/qwen3_6/impl/runtime/logical_kv_store.h` |
| 页地址核 | 核里从页号算地址 | `src/ops/kernel/paged_kv_address.cuh` |

## 6. 卸载水位（主动卸载，不是等池溢出）

`--kv-unload-watermark-pages N`：自由 text-KV 页数降到该水位时，Engine **主动卸载**它判定不可加载的
块（`apps/cli/options.cpp:636`）。

- 默认 `kUnloadWatermarkDerive`（`apps/cli/options.h:128`）：从计划的预填充单元推；**`0` = 关**。
- 优先序 `CLI > env > default`，环境侧解析在 `apps/cli/options.cpp:747-758`，超 u32 域**具名拒绝**（`:752-755`）。
- `apps/cli/options.cpp:741-746` 自述这样做的理由：**被静默忽略的水位，是一次自以为受保护而实际没有保护的运行**。
- 两个前端共用一个 `*_explicit` 门（CLI `:744` 注释指向 `src/serve/serve_options.cpp:836-847`），
  使「flag 是否胜过 env」不可能在两个前端不一致。

## 7. 权重卸载（W13）

| 开关 | 语义 | 出处 |
|---|---|---|
| `--weight-host-bytes N` | 卸载到 host 的字节数 | `apps/cli/options.cpp:657`，默认 0（`apps/cli/options.h:134`） |
| `--weight-device-arena-bytes N` | 设备侧权重 arena | `:659`，默认 0（`apps/cli/options.h:135`） |
| `--weight-prefetch-layers N` | 预取层数，**<2 拒绝** | `:661`，默认 2（`apps/cli/options.h:136`）；拒绝语 `:663-667` 点名「低于 2 会让正在算的层自己的 slot 被自己的预取覆盖」 |
| `--weight-span-floor-bytes N` | 权重跨度下限 | `:668`，默认 0（`apps/cli/options.h:137`） |

主题头：`src/product/weight_offload_budget.h`（8183 B）、`src/product/weight_residency.h`（39739 B）；
测试 `tests/test_weight_residency.cpp`。

`apps/cli/options.h:132-133` 自述这四个开关存在的理由：CLI 与服务端到达**同一批引擎旋钮**，
所以 1M 的运行可以从任一个前端定尺寸。

## 8. KV 容量解析（容量与余量）

| 项 | 语义 | 出处 |
|---|---|---|
| `auto` 的余量 | 留 **1024 MiB** 定尺寸余量 | `include/ninfer/types.h:233`（`kDefaultKvCapacityHeadroomBytes = 1024ULL*1024ULL*1024ULL`）；用法文本 `apps/cli/options.cpp:286-288` 把这个数**现算**进文本 |
| 两种模式 | `Explicit` / `Automatic`，各自的非法组合都拒绝 | `src/targets/registry.cpp:43-60`（显式必须非 0 `:45-47`、显式不得带自动余量 `:48-51`、自动不得带显式 token `:54-57`） |
| 不给 `--kv-capacity` 时 | 用 `explicit_capacity(max_context)` 推 | `apps/cli/options.cpp:839-841` |
