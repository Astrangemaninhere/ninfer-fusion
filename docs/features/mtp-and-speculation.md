# 投机解码：MTP / DFlash / DFlash2 / DSpark

## 1. 后端枚举与拼写

| 项 | 内容 | 出处 |
|---|---|---|
| 引擎侧枚举 | `None / Mtp / DFlash / DFlash2 / Auto` | `include/ninfer/types.h:262-268` |
| 名字表 | `none`/`off`、`mtp`、`dflash`、`dflash2`、`dspark`、`auto` | `src/product/speculative_options.h:11-45` |
| CLI 默认 | **`Auto`** | `apps/cli/options.h:42`（`speculative{SpeculativeBackend::Auto}`）；`apps/cli/options.cpp` 用法文本 `:223` 自述「--spec defaults to auto; none turns speculation off」 |
| serve 默认 | 同样 `Auto`（`src/product/speculative_options.h:12-13` 自述两个前端都默认 auto，所以必须有一个「关回去」的拼写） | 同上 |

### `dspark` 不是第四个后端

`--spec dspark` 解析**就是** `DFlash`（`src/product/speculative_options.h:42`）。究竟跑不跑 Markov 头由
**artifact 自己的 `weights_id`** 决定，证据链写在 `src/product/speculative_options.h:18-41` 的注释里：
DFlash 草稿步在 `markov_w1`/`markov_w2` 绑定时才调 `ops::dspark_markov_argmax`，否则退回 `ops::argmax`；
那两个权重只为 `WeightsProfile::Qwen38Nvfp4Dspark`（`weights_id="nvfp4-dspark"`）物化。

⇒ 文档里可以写「`--spec dspark` 是一个拼写」，**不能**写「它保证加载的 artifact 带 Markov 头」——
`src/product/speculative_options.h:38-41` 明说「接受这个拼写并不承诺这件事」，要在没有该头的制品上拒绝它
需要制品（target package 的 `resolve_weights`/`plan_load`），这一个纯枚举翻译做不到。

## 2. 草稿宽度：三种形态

| 形态 | 拼写 | 行为 |
|---|---|---|
| 钉住一个宽度 | `--spec mtp --draft-tokens k > 0` | 与加梯子之前的固定 k 行为**逐位相同**（`src/product/speculative_options.h:79-80` 自述「every recorded fixed-k experiment」保持行为） |
| **自适应阶梯** | `--spec mtp --draft-tokens 0`、或宽度未给、或 `NINFER_MTP_ADAPTIVE=1` | 引擎捕获宽度梯子，由 `mtp_window_cut` 的**生存/代价准则按轮**选档（`src/product/speculative_options.h:76-78`） |
| **树** | `--spec mtp --draft-tree L,d` | L 条 rank-path × d 步，每节点一个验证列 + anchor；节点预算 `L*d` **就是**这一轮的草稿宽度，所以 `--draft-tree` 与 `--draft-tokens` 是同一个数的两种拼写，不能同时给（`apps/cli/options.cpp:827-837`；用法 `:223`） |

### 域与上界

| 后端 | 域 | 出处 |
|---|---|---|
| `mtp` | `[0,15]`，**0 = 自适应** | `src/product/speculative_options.h:86-89`（拒绝语逐字含「0 = adaptive」） |
| `dflash` | `[1,15]` | `:92-94` |
| `dflash2` | `[1,15]`（0 保留历史的 7 草稿默认） | `:96-100` |
| `none` / `auto` | 不许带 `--draft-tokens` 或非 Full 头 | `:66-69`、`:107-111` |
| 上界的来历 | 5 → 15 是**实测**：该注释自述 on code `k=3/5/9` 给 `196.6/225.7/327.6 tok/s`，而 on Chinese `k=5` 比 `k=3` 差 ⇒ 合适的 k 依赖内容 | `:73-80` |

## 3. 自适应草稿阶梯的机制

| 项 | 出处 |
|---|---|
| 准则与调用点 | `src/targets/qwen3_6/impl/runtime/mtp_window_cut.h`（370 行）；调用点 `src/targets/qwen3_6/impl/runtime/program_impl.h:11073-11128` |
| env 名表 | `NINFER_MTP_WINDOW_CUT`（`:15`）、`NINFER_MTP_WINDOW_TRACE`（`:71`, `:114`）、`NINFER_MTP_WINDOW_RATIO`（`:95`）、`NINFER_MTP_WINDOW_UNOBSERVED`（`:146`）、`NINFER_MTP_WINDOW_DENOM`/`MINREACH`/`SHRINK`（`:267`） |

## 4. MTP 树验证

| 项 | 内容 | 出处 |
|---|---|---|
| 结构常量 | `kMtpDecodeMaximumDrafts = 15`、`kMtpDecodeMaximumWidth = 16`、`kMtpTreeMaximumPaths = kMtpDecodeMaximumDrafts`、`kMtpTreeProposalDepthStride = paths * kMaximumConcurrency`、`kMtpTreeProposalEntries = stride * kMtpDecodeMaximumDrafts` | `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/round_state.h:17,18,22,26,30` |
| 提案帧布局 | `s * kMtpTreeProposalDepthStride + i * <本轮 lane 数> + t` | `round_state.h:175`（数组声明 `:183`） |
| 三个环节分开落地 | `mtp_tree_produce.h`（258 行）、`mtp_tree_proposal_fill.h`（439 行）、`mtp_tree_publish.h`（244 行） | `src/targets/qwen3_6/impl/runtime/`（行数实测 `wc -l`） |
| 契约文档 | `docs/maintainer/mtp-draft-tree.md`（399 行，树内已存在） | 同上 |
| 退化链的一致性检查 | `--draft-tree 1,d` 是退化链，必须**逐 token 复现** `--draft-tokens d`，这就是树路径自己的仪器检查 | 用法 `apps/cli/options.cpp:223` |
| 提案填充的判据 | 把缺陷**当作文本**返回（`mtp_proposal_row_defect()`），三类判不了的情形**就地具名** | `src/targets/qwen3_6/impl/runtime/mtp_tree_proposal_fill.h` |

## 5. DFlash2

| 项 | 内容 | 出处 |
|---|---|---|
| 实现 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h`（596 行） | 行数实测 |
| 特征链落盘 | env `NINFER_DF2FEAT`，目录 `NINFER_DF2FEAT_DIR` | `dflash2_impl.h:146,148,154` |
| 分数表落盘 | env `NINFER_DF2SCORES`，目录 `NINFER_DF2SCORES_DIR` | `dflash2_impl.h:419,421,426` |
| 轮次调试打印 | env `NINFER_DF2DBG` | `src/targets/qwen3_6/impl/runtime/program_impl.h:17002` |
| 树/束走查与 selector | `include/ninfer/ops/dflash2_ddtree.h`（19000 B）、`dflash2_ddtree_beam.h`（23969 B）、`dflash2_tree_walk.h`、`dflash2_selector.h` | `include/ninfer/ops/` 目录实测 |
| 状态文档 | `docs/maintainer/speculative-dflash2-status.md`（73 行） | 树内已存在 |

## 6. 草稿头（proposal head）

| 项 | 内容 | 出处 |
|---|---|---|
| 枚举 | `Full / Optimized / Auto` | `include/ninfer/types.h:251-260` |
| `Auto` 解析在哪里、为什么必须在那里 | `resolved_auto_speculative`：DFlash2 制品带 `text/draft_head` ⇒ `Optimized`，其余留 `Full`；**必须在那一处解析**，因为它的结果同时喂 planner、load plan 与 program，否则冻结的启动特性检查会拒掉加载的权重 | `include/ninfer/types.h:254-258`；实现 `src/targets/registry.cpp:101-112` |
| `--spec auto` 的提前解析 | 同上，且注释逐字说明此前只有 `plan_load` 解析它，导致 planner 建出 Auto 计划而被冻结检查拒绝 | `src/targets/registry.cpp:107-110` |
| 短名单草稿头 | `--lm-head-draft` / `--no-lm-head-draft` | `apps/cli/options.cpp:598,678` |
| 头行的转换器侧 | `tools/convert/qwen3_6/common/draft_head.py`（168 行）、`tools/convert/qwen3_6_27b/draft_head.py`（76 行）、`tools/convert/qwen3_6_35b_a3b/draft_head.py`（67 行）、`tools/convert/qwen3_8_27b/mtp.py`（475 行）、`tools/convert/qwen3_8_flash_next/splice_mtp.py`（190 行） | 行数实测 `wc -l` |

## 7. 接受率与验证的仪器

| env | 作用 | 出处 |
|---|---|---|
| `NINFER_ACCEPTLOG` | 每个 MTP/DFlash 验证轮打一个明细块；**惰性**（不设不打） | `src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:18`、判定 `:67`、打印点 `:193`；`dflash2_impl.h:592`；`mtp_impl.h:434`；`schedule.h:210` |
| `NINFER_ACCEPTLOG_STREAM_SYNC` | 默认开；`=0` 恢复旧行为。该文件自述块内的次序**就是修复本身** | `speculative_target_impl.h:70,75,205,208` |
| `NINFER_SVIP_THRESHOLD` / `NINFER_DFLASH_SVIP_THRESHOLD` | SVIP 阈值 | `src/targets/qwen3_6/impl/runtime/mtp_impl.h:282`；`program_impl.h:715`（`=0` 关掉上限，`program_impl.h:712` 自述） |
| `NINFER_ADAPTIVE_WINDOW` | 自适应窗 | `mtp_impl.h:289` |
| `NINFER_DF2SEL` / `NINFER_DF2_PAIR_SCALE` | selector 调试与配对尺度 | `src/ops/launcher/dflash2_selector.cu:53` / `:28` |

## 8. MTP 的 per-variant 域（CLI 侧仍是硬编码，见未完成册）

| 族 | 域常量所在 | 值 |
|---|---|---|
| `qwen3_6_27b` | `src/targets/qwen3_6_27b/impl/variant.h:37` | `kMaximumMtpDraftTokens` |
| `qwen3_6_35b_a3b` | `src/targets/qwen3_6_35b_a3b/impl/variant.h:33` | 同上 |
| `muse_glimmer_30b` | `src/targets/muse_glimmer_30b/impl/variant.h:39` | 同上 |
| `qwen3_5_9b` | `src/targets/qwen3_5_9b/impl/variant.h:47` | 同上 |
| 统一入口 | `src/targets/qwen3_6/impl/runtime/instance.h:62`（`kMaximumMtpDraftTokens = Variant::maximum_mtp_draft_tokens`） | — |

⚠ 上一轮清单把这几处的行号记在 `impl/config.h`；实测字段在 `impl/variant.h`（同类字段还有
`maximum_dflash_draft_tokens`、`supports_dflash`、`supports_dflash2`、`draft_head_rows`，
见 `src/targets/qwen3_6_27b/impl/variant.h:37-42`）。
