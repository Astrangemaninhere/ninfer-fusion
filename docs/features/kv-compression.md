# KV 压缩与混合精度

KV 是这棵树里新增密度高的地方：`src/product/` 下**19 个 `kv_*.h` 主题头**
（实测 `wc -l src/product/kv_*.h`），另加 `src/kvcfg/`
两个文件（`kv_formats.h` 23388 B / `kv_formats_test.cpp` 5691 B）。

## 1. 档位这个词有三层，别混

| 层 | 是什么 | 出处 |
|---|---|---|
| CLI 全局档 | `bf16`、`int8`、`fp8`、`nvfp4`、`iso4e`（别名 `iso3`）、`rk4v4`（别名 `e8`） | `apps/cli/options.cpp:93-107`；用法表 `:159` |
| 引擎侧词汇表枚举 | `Auto`、`Bf16`、`Fp16`、`Int8`、`Int4`、`Iso4`、`Iso4e`、`Rk4v4`、`Rk3v4`、`Rk2v4` | `src/kvcfg/kv_formats.h:27-66` |
| DType（census 的判据） | 两个不同拼写可以映射到**同一个** DType，此时它们算**一个** codec | `src/product/kv_plane_census.h:43-49` |

引擎侧词汇表的头注释给出了两个关键事实：

1. **`Iso4e` 落地的码字是 4bit 符号-幅值 nibble**（bit3 符号 + 幅值 0..7，组 16，E4M3 scale = amax/7，
   两码/字节），不是参考契约里那个「符号+2bit 幅值、8 元素/3 字节」的 3bit 形式 —— 后者**没有实现**，
   所以 `bits_of` 按实际的 4 记（`src/kvcfg/kv_formats.h:33-42` 自述，并自陈「原来的 3 是拿参考契约当实现，
   会误导 `--kv-tier-formats` 的精度排序判断」）。
2. **`Rk3v4` / `Rk2v4` 现在是真 E8 格点**：几何 6656 B / 4608 B per head-page，层成本
   4.25 → 3.75 → 3.25；这两行的码字是 E8 ±1/4 的 8 维格点（2bit 16 位 / 3bit 24 位），
   由 `e8_kv_plane_codec_of_record()` 返回 `Lattice`；标量 codec 保留为对照臂
   （`src/kvcfg/kv_formats.h:44-65`）。`Rk4v4` 仍是标量行，理由是格点码字没有 32 位形式。

## 2. 逐层存储表与混合 store

| 功能 | 是什么 | 出处 |
|---|---|---|
| 逐层策略头 | 每层一个存储档的主题头，不是散在 planner 里 | `src/product/kv_perlayer_policy.h`（842 行） |
| **plane census** | 统计一个 KV 表里有几种 distinct codec（上限 **8**），判「这是不是一个混合 store」，并渲染成一行 | `src/product/kv_plane_census.h:99`（`kKvPlaneCensusCodecs = 8`）、`:101`（`struct KvPlaneCensus`）、`:121`（`kv_plane_census`）、`:180`（内存摘要重载）、`:197`（`kv_plane_census_line`） |
| 存储档 → DType 映射 | 唯一决策点 `product::kv_dtype_for_storage` | `src/product/kv_storage_dtype.h`（363 行）；census 在 `src/product/kv_plane_census.h:51-56` 委托给它 |
| 平面丢弃 | 整层不建 KV plane | env `NINFER_KV_DROP_LAYERS`；解析 `src/targets/qwen3_6/impl/state/decoder_state.cpp:221`（`parse_kv_layer_drop`） |
| 组件开关 | KV 三个组件开关（旋转/行尺度/…）的主题头 | `src/product/kv_component_switch.h`（344 行） |
| 摘要格式 | KV 摘要的渲染格式头 | `src/product/kv_summary_format.h`（165 行，mtime 2026-09-20 16:36） |
| 召回块 | KV 块语义目录与召回 | `src/product/kv_recall_block.h`（451 行） |
| e8 宽度族 | e8 lattice 的位宽推导，被 `static_assert` 钉住 | `src/product/kv_e8_width.h`（194 行）、`src/product/kv_e8_width_codec.h`（274 行）；出厂行 425/8704 的复现要求写在 `src/kvcfg/kv_formats.h:49-50` |

**census 拒绝的三种松读法**（这是它的存在理由，`src/product/kv_plane_census.h` 头注释逐条给出）：

- 把「被丢弃的层」算成一种 codec：那会把只有一个 codec 但有洞的 store 报成
  `{dropped, nvfp4}` 的**混合** store，即凭空造出两 codec 页（`:28-32`，由具名分支拒绝而不是靠循环顺序）。
- 按存储档拼写而不是按 DType 计数：`Fp8E4M3Row256` 与 `Fp8Group16` 都落到 `DType::FP8_E4M3FN`，
  按拼写计会把它报成混合（`:43-49`）。
- 把引擎建不出来的 codec 计进去：所以每个带 plane 的 codec 必须可解析，不可解析的**继承**
  `kv_dtype_for_storage` 的拒绝而不是另发明一个（`:51-56`）。
- 输入是**已解析的逐层表**而不是选项字符串，所以 census 说的是「这次运行」，不是「要求了什么」：
  `--kv-dtype nvfp4` 之上再叠一张注册的默认表就是一个混合 store，而单看全局档说不出这件事（`:36-41`）。

## 3. 位预算求解器

| 功能 | 是什么 | 出处 |
|---|---|---|
| 位预算拟合器 | 在声明上限内压低罚分，解出逐层档位表；候选与罚分表由 `product::kv_gear_candidate_list()` 导出 | `src/product/kv_bit_budget.h`（2490 行）；候选表**拼进**用法文本而非手写（`apps/cli/options.cpp:183-184`） |
| K/V 位宽三读法 | `joint`（一个总上限）/ `split`（K、V 各自逐层分层）/ `ceiling`（故意部署 `min(k,v)` 并报出花不掉的余量） | `src/product/kv_kv_bits.h`（1341 行）；用法语义全文 `apps/cli/options.cpp:169-179` |
| 某层落不到同一个档时 | **按索引具名拒绝**并打印 `min(k,v)` 的可部署方案，使「改一个旗标就能跑」 | 用法 `apps/cli/options.cpp:173-176` |
| 适配求解器 | 用户给的 K/V 档位 → 逐层可部署存储方案 | `src/product/kv_adapt_solver.h`（619 行）；离线侧 `tools/archkit/kv_auto_allocate.py`（132 行） |
| 离线镜像 | 位预算求解的离线侧，与服务端结果交叉核对 | `tools/archkit/kv_budget_mirror.py`（510 行）、`kv_budget_probe.cpp`（242 行）、`kv_budget_regen.sh`（179 行）、`kv_bit_budget.py`（154 行） |
| 同 bit 换 codec | `--kv-codec-preference`：在**同 bit 成本**的候选里挑哪一种 | 用法给出可复算的例子：`--kv-bits 4.5 --kv-quality-weight 0 --kv-codec-preference iso4e` 返回 `0-15:iso4e`，同一命令去掉偏好返回 `0-15:nvfp4`（`apps/cli/options.cpp:193-199`） |

## 4. 行尺度校准闭环（kvrowscale）

`--kv-row-scale auto` 是**唯一**跑闭环的态（`apps/cli/options.h:103` 默认空串）。

| 功能 | 是什么 | 出处 |
|---|---|---|
| 闭环主流程 | 表存在就加载并跳过采集；没有、或为**别的模型/别的 KV 配置**烤的就本次采集一次并写一张 | `src/product/kv_rowscale_persist.h`（939 行）；后缀常量 `:95`（`.kvrowscale.bin`）；主流程 `:480` 起；命中已落盘表 `:697`；异配置移开 `:758-762`；**图开启时拒绝** `:770` |
| 四个伴生文件 | 表本体 + 采集记录 + 「无需校准」原因 + 被拒的旧表 | `src/product/kv_rowscale_persist.h:28-40` |
| 表格式 | 11 字节魔数 `NINFERKVRS1` | `:452`（`std::memcpy(out, "NINFERKVRS1", 11)`） |
| 帧身份戳 | 把 `.kvc` 帧头 offset 48..63 那 16 个保留字节赋予含义：`config_fingerprint`(8) + `producer_version`(4) + `flags`(4) | `src/product/kv_rowscale_frame_identity.h`（133 行）；偏移常量 `:44-51`；env `NINFER_KV_ROWSCALE_IDENTITY` 在 `:77`；**全零是拒绝不是通配**（`:28-33`） |
| rope 域进指纹 | `--yarn` 会改 K 经过的 rope 域，所以它进行尺度指纹 `rope_regime` | `src/product/kv_rowscale_persist.h:151` 起；用法 `apps/cli/options.cpp:249-253` 自述「没点 `--yarn` 烤的表，不能被点了 `--yarn` 的运行采用」 |
| 首次校准的前置 | 必须配 `--no-cuda-graph` | 用法 `apps/cli/options.cpp:242`（「The first calibration run needs --no-cuda-graph」） |
| 落盘工具 | 把 solve 结果烤成表 | `src/product/kv_rowscale_bake.h`（597 行）；`tools/kv_rowscale_sidecar.py` |

## 5. 由环境变量进入的两项

| 开关 | 语义 | 读取点 |
|---|---|---|
| `NINFER_KV_PAGING_PREALLOC` | KV 页池预分配闸：非空且非 `"0"` 时做容量可行性检查 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2093`；头 `src/product/kv_paging_preallocation.h`（582 行） |
| `NINFER_KV_DROP_LAYERS` | `"0,3,7"` 或 `"2-5"`；丢的层不推任何 KV plane | 解析 `src/targets/qwen3_6/impl/state/decoder_state.cpp:221`；语法与 `--kv-residual-layers` 同一个（`src/product/kv_options.h:151`）；丢层在 `src/targets/qwen3_6/impl/runtime/layouts_impl.h:137` 因「无存储剖面」被具名拒绝；另外三处读取 `apps/perplexity/main.cpp:529,531`、`src/core/device_capabilities.h:306` |

## 6. 相关契约文档（树内已存在，行数实测）

`docs/maintainer/kv-strategy-matrix.md`（150 行）、`docs/maintainer/kv-storage-names-and-switch-gates.md`
（273 行）、`docs/maintainer/paged-kv-cache.md`（736 行）、`docs/maintainer/tensor-formats.md`（863 行）、
`docs/maintainer/storage-layouts.md`（357 行）。

## 7. 本册没写进去的

- `--kv-tail-tokens` 与它代表的「近期高精度窗」档位：**按名找不到**，见
  [`unfinished.md`](unfinished.md) 第 ?1 条。
- `tail` 档当前只能重复 `hot` 的值：见 [`unfinished.md`](unfinished.md) 第 A17 条。
