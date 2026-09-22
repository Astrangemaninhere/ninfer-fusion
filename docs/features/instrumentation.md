# 仪器与诊断环境变量

本册的**全部**条目来自一次机械普查：`grep -rn 'getenv("NINFER' src/ apps/ include/ tools/`
（排除 `__pycache__` 与 `*.pre_land` / `*.gb2-orig` 前像文件）。逐条 `file:line` 由该命令直接给出。

⚠ 三个读法上的坑，先说清：

1. **`*.pre_land` / `*.gb2-orig` 是前像文件，不是活代码**。同一批 env 在两处出现时，活的是不带后缀的那份
   （例：`program_impl.h:11073` 活，`program_impl.h.pre_land:11012` 是前像）。
2. **编译期宏不是环境变量**。`NINFER_BUILD_CUDA_ARCHS`、`NINFER_HAVE_QPN`、
   `NINFER_ARCH_SIM_ENABLED`、`NINFER_MMA_HAS_*` 等只能 `#ifdef` 观测，`getenv` 一个都测不到。
   `NINFER_ARCH_WARN` 是**真的环境变量**（`src/core/arch_caps.h:1718`），别因为它名字像宏就当成宏。
3. **归族不是按名字前缀猜的**，是按读取点所在目录归的。

## 1. 用户面：与旗标成对的开关

| env | 语义 | 读取点 |
|---|---|---|
| `NINFER_FT_BW_GOV` | `--prefill-chunk-mode` 的 env 拼写：`0` = manual，`1` = dynamic；**flag 胜过 env** | 解析一次的唯一点 `src/runtime/engine/engine.cpp:86`（说明文本 `src/runtime/engine/bandwidth_governor.h:24,27`） |
| `NINFER_FT_BW_TRACE` | `=1` 打印治理器模式与它实际装的单元 | `src/runtime/engine/bandwidth_governor.h:321` |
| `NINFER_FT_STATS` | `--ft-stats` 的 env 拼写；FreeToken 逐层注意力能量观测总开关 | `src/ops/common/ft_stats.h:33`、`src/targets/qwen3_6/impl/runtime/ft_stats.h:31`；parse 期写入点 `apps/cli/options.cpp:966-969` |
| `NINFER_FT_PERIOD` | 观测周期 | `src/ops/common/ft_stats.h:41`、`src/targets/qwen3_6/impl/runtime/ft_stats.h:39` |
| `NINFER_KV_UNLOAD_WATERMARK_PAGES` | 卸载水位 | `apps/cli/options.cpp:748`、`src/serve/serve_options.cpp:905` |
| `NINFER_MTP_ADAPTIVE` | 临时逃生舱（见 [未完成册](unfinished.md) 第 B9 条） | `apps/cli/options.cpp:796`、`src/serve/serve_options.cpp:697` |
| `NINFER_KV_ROWSCALE` | `--kv-row-scale` 的 env 等价 | `src/ops/kernel/gqa_isoquant_row_scale_loader.cu:22`、`src/targets/qwen3_6/impl/state/decoder_state.cpp:1099` |
| `NINFER_KV_ROTATION` | `--kv-rotation` 的核侧读取 | `src/ops/kernel/gqa_isoquant_rot.cu:117`、`src/targets/qwen3_6/impl/state/decoder_state.cpp:1168` |
| `NINFER_KV_DROP_LAYERS` | 平面丢弃 | `src/targets/qwen3_6/impl/state/decoder_state.cpp`（解析函数 `:221`）、`apps/perplexity/main.cpp:529,531` |
| `NINFER_KV_PAGING_PREALLOC` | 页池预分配闸（opt-in） | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:2093` |
| `NINFER_KV_CALIB_DIR` | 校准目录 | `src/product/kv_rowscale_persist.h:816`、`src/targets/qwen3_6/impl/runtime/text_context_impl.h:137,145` |
| `NINFER_KV_CALIB_MAX_TOKENS` | 校准采集的 token 上限 | `src/targets/qwen3_6/impl/runtime/kv_calibration.h:195`（说明文本 `src/product/kv_rowscale_persist.h:835` 给出 4096 的默认值） |
| `NINFER_KV_ROWSCALE_IDENTITY` | 校准帧生产者身份，`"<12 hex>:<version>"` | 常量 `src/product/kv_rowscale_frame_identity.h:77` |

## 2. 服务端重排族（FreeToken 观测 → 周期重排）

| env | 语义 | 读取点 |
|---|---|---|
| `NINFER_FT_RELOAD_SECS` | 重排周期 | `src/serve/kv_auto_relayout.cpp:241` |
| `NINFER_FT_FULL_ATTN_LAYERS` | 要覆盖的全注意力层 | `src/serve/kv_auto_relayout.cpp:245` |
| `NINFER_FT_DEEP_FRAC` | 「深」层占比 | `src/serve/kv_auto_relayout.cpp:249` |
| `NINFER_FT_VRAM_AXIS` | VRAM 轴开关 | `src/serve/kv_auto_relayout.cpp:253` |
| `NINFER_FT_COLD_PAGES` | 冷页数 | `src/serve/kv_auto_relayout.cpp:379` |

## 3. MTP / 投机族

| env | 语义 | 读取点 |
|---|---|---|
| `NINFER_MTP_WINDOW_CUT` | 阶梯开关 | `src/targets/qwen3_6/impl/runtime/program_impl.h:11073` |
| `NINFER_MTP_WINDOW_RATIO` | 比值覆盖 | `program_impl.h:11089` |
| `NINFER_MTP_WINDOW_UNOBSERVED` | 未观测层的定价 | `program_impl.h:11101` |
| `NINFER_MTP_WINDOW_DENOM` | 分母 | `program_impl.h:11109` |
| `NINFER_MTP_WINDOW_TRACE` | 追踪 | `program_impl.h:11117` |
| `NINFER_MTP_WINDOW_SHRINK` | 收缩 | `program_impl.h:11124` |
| `NINFER_MTP_WINDOW_MINREACH` | 可达下限 | `program_impl.h:11128` |
| `NINFER_ACCEPTLOG` | 每验证轮一个明细块 | `src/targets/qwen3_6/impl/runtime/speculative_target_impl.h:67,193` |
| `NINFER_ACCEPTLOG_STREAM_SYNC` | 默认开；`=0` 恢复旧行为 | `speculative_target_impl.h:75` |
| `NINFER_SVIP_THRESHOLD` | SVIP 阈值 | `src/targets/qwen3_6/impl/runtime/mtp_impl.h:282` |
| `NINFER_ADAPTIVE_WINDOW` | 自适应窗 | `mtp_impl.h:289` |
| `NINFER_DFLASH_SVIP_THRESHOLD` | DFlash 侧的同一旋钮（`=0` 关掉上限） | `program_impl.h:715`，说明 `:712` |
| `NINFER_DF2FEAT` / `NINFER_DF2FEAT_DIR` | DFlash2 特征链落盘与目录 | `src/targets/qwen3_6/impl/runtime/dflash2_impl.h:148,154` |
| `NINFER_DF2SCORES` / `NINFER_DF2SCORES_DIR` | 分数表落盘与目录 | `dflash2_impl.h:421,426` |
| `NINFER_DF2DBG` | 轮次调试打印 | `program_impl.h:17002` |
| `NINFER_DF2SEL` / `NINFER_DF2_PAIR_SCALE` | selector 调试 / 配对尺度 | `src/ops/launcher/dflash2_selector.cu:53` / `:28` |

## 4. 召回 / 目录 / 文本追回族

| env | 读取点 |
|---|---|
| `NINFER_TURN_RECALL` / `NINFER_TURN_RECALL_FSYNC` / `NINFER_TURN_RECALL_BYTES` | `program_impl.h:1167,1192,1213,1235` |
| `NINFER_RECALL_TEXT` / `NINFER_RECALL_TEXT_REBUILD` | `program_impl.h:1312,13733,13805`；另有 `src/runtime/engine/engine.cpp:131-132` |
| `NINFER_RECALL_TIMING` / `NINFER_RECALL_REACH` | `program_impl.h:14225,14674` |
| `NINFER_SUM_DIR` / `_IDLE_SECS` / `_BLOCKS` / `_BYTE_AXIS` | `src/spec/sum_dir.h:730,733,737,741`；`program_impl.h:12942` |
| `NINFER_SUM_DIR_VEC` / `_DIM` / `_TOPK` / `_METRIC` / `_MISSING` / `_OFFER` | `src/spec/sum_dir_vector.h:183,186,192,196,199,202` |

## 5. 布局 / 权重 / 观测族

| env | 读取点 |
|---|---|
| `NINFER_WS_DUMP` / `NINFER_WS_HEADROOM_PCT` | `src/targets/qwen3_6/impl/runtime/layouts_impl.h:1062` / `:1055` |
| `NINFER_PROBE_REPORT` | `layouts_impl.h:1329` |
| `NINFER_KV_WINDOW_TOKENS` | `layouts_impl.h:298` |
| `NINFER_KVDUMP_DIR` / `NINFER_KVDUMP_LAYERS` / `NINFER_KVDUMP_PAGES` | `src/targets/qwen3_6/impl/runtime/text_context_impl.h:107`；`text_prefill_impl.h:225,242` |
| `NINFER_HEADDBG` | `text_context_impl.h:59` |
| `NINFER_HS_DUMP_DIR` / `NINFER_HS_DUMP_TOPK` | `text_prefill_impl.h:102,150,333` |
| `NINFER_PLE_STATS` | `src/ops/ple/ple_stage.h:146` |
| `NINFER_GENERIC_ROWDEC` | `src/ops/generic/rowsplit_generic.cu:377` |
| `NINFER_ATTENTION_SIMT_FFMA` | `src/ops/kernel/gqa_attention_simt_ffma.cuh:298` |
| `NINFER_ARCH_WARN` | `src/core/arch_caps.h:1718` |
| `NINFER_ARENA_TRACE` | `src/core/arena.cu:24` |
| `NINFER_PDL_TRACE` | `src/core/pdl.cuh:93` |
| `NINFER_FAULT_INJECT` / `NINFER_FAULT_INJECT_MIN_ID` | `src/runtime/engine/engine_core.h:2123,2128` |
| `NINFER_REGISTRY_DUMP` / `NINFER_EXPORT_HEAD_DIR` | `src/targets/registry.cpp:489` / `:189` |
| `NINFER_GFX906_TP2_FLAG_SYNC` | `src/ops/common/allreduce.cu:263`；说明 `include/ninfer/ops/allreduce.h:144` |

## 6. 验证 / 分片调试族（多为核内一次性打印）

| env | 读取点 |
|---|---|
| `NINFER_VERIFY_EXACT` | `src/ops/kernel/gqa_attention_decode.cuh:69`、`src/ops/launcher/gqa_attention_decode_e8.cu:54`、`src/ops/launcher/gqa_attention_decode_partial.cuh:88`、`src/ops/launcher/gqa_attention_decode_smallt.cu:83` |
| `NINFER_SPLITDBG` / `NINFER_SPLITDBG_MAX` | `gqa_attention_decode_e8.cu:28,38`、`gqa_attention_decode_partial.cuh:58,72`、`gqa_attention_decode_smallt.cu:27,37` |
| `NINFER_SPLIT_PARITY` / `NINFER_SPLIT_PARITY_WIDTH` | `gqa_attention_decode_smallt.cu:59,67` |
| `NINFER_GQA_SINGLE_WAVE` | `src/ops/kernel/gqa_attention_decode.cuh:129` |

## 7. DFlash2 / FlashNext 目标专属族

`src/targets/qwen3_8_flash_next/` 下有一整族带 `NINFER_FLASH_NEXT_` 前缀的运行期开关，逐条读取点：

| env | 读取点 |
|---|---|
| `NINFER_FLASH_NEXT_HYPER_LEGACY` | `impl/hyper_connection_kernels.cu:723` |
| `NINFER_FLASH_NEXT_MOE_DOWN_LEGACY` | `impl/moe_kernels.cu:1879` |
| `NINFER_FLASH_NEXT_MOE_STAGING` | `impl/moe_kernels.cu:1998` |
| `NINFER_FLASH_NEXT_MOE_SHARED_MMA` | `impl/moe_shared_kernels.cu:250` |
| `NINFER_FLASH_NEXT_ROUTE_LEGACY` | `impl/moe_route.cu:522` |
| `NINFER_FLASH_NEXT_TRACE_ROUTING` | `impl/moe.cpp:85` |
| `NINFER_FLASH_NEXT_QSA_MMA_SCHED` | `impl/qsa_attention_kernels.cu:827` |
| `NINFER_FLASH_NEXT_QSA_PREFILL_MMA` | `impl/runtime_plan.cpp:297` |
| `NINFER_FLASH_NEXT_PREFILL_HOST_SYNC` | `impl/qsa_indexer_kernels.cu:746` |
| `NINFER_FLASH_NEXT_STAGE_LEDGER` | `impl/stage_ledger.cpp:15` |
| `NINFER_FLASH_NEXT_TRACE_STAGES` | `impl/text_executor.cpp:689` |
| `NINFER_FLASH_NEXT_TRACE_ADMISSION` | `impl/program.cpp:1655` |
| `NINFER_FLASH_NEXT_TRACE_KEYS` | `impl/program.cpp:452,2405,2943,2987` |
| `NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS` | `impl/package.cpp:167,245` |
| `NINFER_RANKING_PATH` | `impl/load/materialized.cpp:31` |

## 8. 工具侧（不属于引擎运行时）

`tools/tp2/parity.cpp` 读 `NINFER_TP2_PARITY_OUTPUT`（`:1401`）、`NINFER_TP2_PARITY_DUMP_DIR`（`:397`）、
`NINFER_TP2_PARITY_KV`（`:691`）、`NINFER_QWEN3_8_27B_WEIGHTS`（`:1561`）。

## 9. 编译期宏（**不是**环境变量，容易写错）

`src/CMakeLists.txt` 导出的一批：`NINFER_BUILD_CUDA_ARCHS`（由 `CMAKE_CUDA_ARCHITECTURES` 导出，
因为 `-a` 后缀如 `120a` 无法从 `cudaDeviceProp` 观测）、`NINFER_HAVE_QPN`（9 处）、
`NINFER_BUILD_APPS` / `_SERVE` / `_MEDIA_ACQUIRE` / `_PROMPT_INPUT` / `_ARCHS`、
`NINFER_ENABLE_ARCH_SIM` / `NINFER_ARCH_SIM_ENABLED` / `NINFER_SIM_ARCH` / `NINFER_SIM_ARCH_ACK`、
`NINFER_ENABLE_VENDOR_SIM` / `NINFER_VENDOR_SIM_ENABLED` / `NINFER_SIM_VENDOR` / `NINFER_SIM_VENDOR_ACK`、
`NINFER_VIRTUAL_DEVICES*`、`NINFER_MMA_HAS_*`、`NINFER_MEMORY_HAS_CP_ASYNC`、
`NINFER_NVFP4_TMA_DEVICE_ARCH`、`NINFER_PROBE_HAS_*`。

来源：`src/CMakeLists.txt`（`NINFER_BUILD_*` 一组）与 `CMakeLists.txt:253-275`（两个模拟选项）。
